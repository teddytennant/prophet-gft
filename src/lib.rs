//! Threshold algorithms for prophet gain-from-trade.
//!
//! Arrival-order rules from <https://arxiv.org/abs/2609.30742>. The mechanism
//! holds at most one item. Gain from a completed trade is buyer price minus
//! seller price; unsold inventory contributes nothing.

/// Buyer or seller in the arrival sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Buyer,
    Seller,
}

/// One arriving agent and their posted price.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Agent {
    pub role: Role,
    pub price: f64,
}

/// `rho(n, m) = (n + 1) / (n + m + 2)`.
///
/// On Uniform[0, 1] the Algorithm 1 threshold is this value, because the
/// quantile function is the identity.
pub fn rho(n: usize, m: usize) -> f64 {
    (n as f64 + 1.0) / (n as f64 + m as f64 + 2.0)
}

/// Closed form `R(n) = 2 n (n+3)^n / ((n-1)(n+3)^n + (n+1)^{n+1})` for `n >= 1`.
///
/// `n = 1` makes the `(n-1)` term zero, so `R(1) = 2`.
pub fn r(n: usize) -> f64 {
    let n_f = n as f64;
    let n_i = n as i32;
    let pow_n3 = (n_f + 3.0).powi(n_i);
    let pow_n1 = (n_f + 1.0).powi(n_i + 1);
    (2.0 * n_f * pow_n3) / ((n_f - 1.0) * pow_n3 + pow_n1)
}

/// Balanced-market upper bound
/// `(2 n^2 - n) / ((n + 4^{-n} - 1) (n + 1))`.
pub fn balanced_bound(n: usize) -> f64 {
    let n_f = n as f64;
    let num = 2.0 * n_f * n_f - n_f;
    let den = (n_f + 4.0_f64.powf(-(n as f64)) - 1.0) * (n_f + 1.0);
    num / den
}

/// Lemma 4.3: `(q+1)^q / ((q+1)^q - q^q)` for `q >= 1`.
pub fn lemma43(q: usize) -> f64 {
    let q_f = q as f64;
    let q_i = q as i32;
    let a = (q_f + 1.0).powi(q_i);
    let b = q_f.powi(q_i);
    a / (a - b)
}

/// Quantile function of Uniform[0, 1].
pub fn uniform_quantile(p: f64) -> f64 {
    p
}

/// Algorithm 1. `n` is the number of buyers. `threshold` is given.
///
/// A seller is bought only when the mechanism is empty, fewer than `n` buyers
/// have been seen, and `price < threshold`. A buyer increments the buyer
/// count, then is sold to only when an item is held and either `price > threshold`
/// or this is the last buyer (`c == n`), which force-sells. `price == threshold`
/// does not trade, except that force-sell.
pub fn algorithm1(agents: &[Agent], n: usize, threshold: f64) -> f64 {
    let mut holding = 0u8;
    let mut buyers_seen = 0usize;
    let mut seller_price = 0.0;
    let mut gft = 0.0;

    for agent in agents {
        match agent.role {
            Role::Seller => {
                if holding == 0 && buyers_seen < n && agent.price < threshold {
                    seller_price = agent.price;
                    holding = 1;
                }
            }
            Role::Buyer => {
                buyers_seen += 1;
                if holding == 1 && (agent.price > threshold || buyers_seen == n) {
                    gft += agent.price - seller_price;
                    holding = 0;
                }
            }
        }
    }

    gft
}

/// Algorithm 2, single seller (`m = 1`). `agents` is `n` buyers plus one seller.
///
/// `T` depends on the seller's 1-based position `k`:
/// if `k == n + 1` the seller is last and gain is 0; otherwise
/// `q = n - k + 1` and `T = quantile(q / (q + 1))`. The seller is bought only
/// when `price < T`. Later buyers who are not the last agent are sold to at
/// the first price strictly above `T`. If none qualifies, the item is
/// force-sold to the agent at position `n + 1`.
pub fn algorithm2<F>(agents: &[Agent], n: usize, quantile: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let Some(seller_idx) = agents.iter().position(|a| a.role == Role::Seller) else {
        return 0.0;
    };
    let k = seller_idx + 1;
    if k >= n + 1 {
        return 0.0;
    }
    let q = n - k + 1;
    let threshold = quantile((q as f64) / (q as f64 + 1.0));
    let seller_price = agents[seller_idx].price;
    if seller_price >= threshold {
        return 0.0;
    }

    for (idx, agent) in agents.iter().enumerate().skip(seller_idx + 1) {
        if agent.role != Role::Buyer {
            continue;
        }
        let position = idx + 1;
        if position == n + 1 {
            return agent.price - seller_price;
        }
        if position < n + 1 && agent.price > threshold {
            return agent.price - seller_price;
        }
    }
    0.0
}

/// Single-seller ex-post optimum on a fixed arrival.
///
/// If any buyer arrives after the seller, this is
/// `max(0, max(later buyer prices) - seller price)`; otherwise 0.
pub fn single_seller_opt(agents: &[Agent]) -> f64 {
    let Some(seller_idx) = agents.iter().position(|a| a.role == Role::Seller) else {
        return 0.0;
    };
    let seller_price = agents[seller_idx].price;
    let mut best_later: Option<f64> = None;
    for agent in agents.iter().skip(seller_idx + 1) {
        if agent.role == Role::Buyer {
            best_later = Some(match best_later {
                Some(best) => best.max(agent.price),
                None => agent.price,
            });
        }
    }
    match best_later {
        Some(price) => (price - seller_price).max(0.0),
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        let diff = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs());
        assert!(
            diff <= 1e-6 || diff <= 1e-9 * scale,
            "actual={actual} expected={expected} abs_diff={diff}"
        );
    }

    fn buyer(price: f64) -> Agent {
        Agent {
            role: Role::Buyer,
            price,
        }
    }

    fn seller(price: f64) -> Agent {
        Agent {
            role: Role::Seller,
            price,
        }
    }

    /// xorshift64. No external RNG crate.
    struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self {
                state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
            }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        fn next_f64(&mut self) -> f64 {
            let bits = self.next_u64() >> 11;
            (bits as f64) * (1.0 / ((1u64 << 53) as f64))
        }

        fn gen_index(&mut self, n: usize) -> usize {
            (self.next_u64() as usize) % n
        }
    }

    fn shuffle(agents: &mut [Agent], rng: &mut XorShift64) {
        for i in (1..agents.len()).rev() {
            let j = rng.gen_index(i + 1);
            agents.swap(i, j);
        }
    }

    fn random_market(rng: &mut XorShift64, n: usize, m: usize) -> Vec<Agent> {
        let mut agents = Vec::with_capacity(n + m);
        for _ in 0..n {
            agents.push(buyer(rng.next_f64()));
        }
        for _ in 0..m {
            agents.push(seller(rng.next_f64()));
        }
        shuffle(&mut agents, rng);
        agents
    }

    /// Spec walk with the invariants the random-order test requires.
    fn checked_algorithm1(agents: &[Agent], n: usize, threshold: f64) -> f64 {
        let buyer_count = agents.iter().filter(|a| a.role == Role::Buyer).count();
        assert_eq!(buyer_count, n, "invariant check assumes exactly n buyers");

        let mut holding = 0u8;
        let mut buyers_seen = 0usize;
        let mut seller_price = 0.0;
        let mut gft = 0.0;

        for agent in agents {
            assert!(holding <= 1, "h never exceeds 1");
            match agent.role {
                Role::Seller => {
                    if holding == 0 && buyers_seen < n && agent.price < threshold {
                        assert!(agent.price < threshold, "buys only when price < T");
                        assert!(buyers_seen < n, "buys only when c < n");
                        seller_price = agent.price;
                        holding = 1;
                    }
                }
                Role::Buyer => {
                    buyers_seen += 1;
                    let want_sell = agent.price > threshold || buyers_seen == n;
                    // Sell only while holding. A flat book never sells, including at c == n.
                    if holding == 1 && want_sell {
                        let forced = agent.price <= threshold;
                        if !forced {
                            assert!(
                                agent.price > threshold,
                                "non-forced sells only when price > T"
                            );
                        } else {
                            assert_eq!(buyers_seen, n, "force-sell only at the last buyer");
                        }
                        gft += agent.price - seller_price;
                        holding = 0;
                    }
                }
            }
            assert!(holding <= 1, "h never exceeds 1");
        }

        assert_eq!(holding, 0, "ends at h = 0");
        gft
    }

    #[test]
    fn r_closed_form() {
        assert_close(r(1), 2.0);
        assert_close(r(2), 100.0 / 52.0);
        assert_close(r(2), 1.923076923076923);

        // R(5) from the closed form, expanded by hand: 8^5 = 32768, 6^6 = 46656.
        let r5 = 2.0 * 5.0 * 8.0_f64.powi(5)
            / ((5.0 - 1.0) * 8.0_f64.powi(5) + 6.0_f64.powi(6));
        assert_close(r5, 327680.0 / 177728.0);
        assert_close(r(5), r5);
        assert_close(r(5), 327680.0 / 177728.0);

        // Limit of R(n) is 2 e^2 / (e^2 + 1). n = 20 has not entered the 0.01
        // neighborhood (gap is about 0.0238); n = 50 has.
        let limit = 1.7615941559557649;
        let gap20 = (r(20) - limit).abs();
        assert!(
            gap20 < 0.03,
            "R(20) = {}, gap {} from {}",
            r(20),
            gap20,
            limit
        );
        assert!(
            (r(50) - limit).abs() < 0.01,
            "R(50) = {} should be within 0.01 of {}",
            r(50),
            limit
        );
    }

    #[test]
    fn balanced_bound_values() {
        assert_close(balanced_bound(1), 2.0);
        assert_close(balanced_bound(2), 6.0 / (1.0625 * 3.0));
        assert_close(balanced_bound(2), 1.8823529411764706);
        assert!(balanced_bound(2) < 2.0);
        assert!(balanced_bound(3) < 2.0);
        assert!(balanced_bound(5) < 2.0);
    }

    #[test]
    fn lemma43_values() {
        assert_close(lemma43(1), 2.0);
        assert_close(lemma43(2), 1.8);
        // e / (e - 1). q = 20 is about 0.0229 away; q = 50 is inside 0.01.
        let limit = 1.5819767068693265;
        let gap20 = (lemma43(20) - limit).abs();
        assert!(
            gap20 < 0.03,
            "lemma43(20) = {}, gap {} from {}",
            lemma43(20),
            gap20,
            limit
        );
        assert!(
            (lemma43(50) - limit).abs() < 0.01,
            "lemma43(50) = {}",
            lemma43(50)
        );
    }

    #[test]
    fn uniform_quantile_is_identity_so_thresholds_match() {
        assert_close(uniform_quantile(rho(1, 1)), 0.5);
        assert_close(rho(1, 1), 2.0 / 4.0);
        let q = 3usize;
        assert_close(
            uniform_quantile(q as f64 / (q as f64 + 1.0)),
            q as f64 / (q as f64 + 1.0),
        );
    }

    #[test]
    fn alg1_hand_examples() {
        let n = 1usize;
        let threshold = rho(1, 1);
        assert_close(threshold, 0.5);

        // Seller 0.2 then buyer 0.9: buy, then sell. GFT = 0.7.
        let order = [seller(0.2), buyer(0.9)];
        assert_close(algorithm1(&order, n, threshold), 0.7);

        // Seller 0.8 is not below T. GFT = 0.
        let order = [seller(0.8), buyer(0.9)];
        assert_close(algorithm1(&order, n, threshold), 0.0);

        // Seller 0.2 then buyer 0.4: buy, force-sell because c == n. GFT = 0.2.
        let order = [seller(0.2), buyer(0.4)];
        assert_close(algorithm1(&order, n, threshold), 0.2);

        // Buyer first consumes the only buyer slot, so the later seller is not bought.
        let order = [buyer(0.9), seller(0.2)];
        assert_close(algorithm1(&order, n, threshold), 0.0);
    }

    #[test]
    fn alg2_hand_examples() {
        // Seller first at 0.2, buyer at 0.9. k = 1, q = 1, T = 0.5. Force-sell. GFT = 0.7.
        let order = [seller(0.2), buyer(0.9)];
        assert_close(algorithm2(&order, 1, uniform_quantile), 0.7);
        assert_close(single_seller_opt(&order), 0.7);

        // Seller first at 0.6 >= T. No buy. GFT = 0.
        let order = [seller(0.6), buyer(0.9)];
        assert_close(algorithm2(&order, 1, uniform_quantile), 0.0);
        assert_close(single_seller_opt(&order), 0.3);

        // Seller last. GFT = 0.
        let order = [buyer(0.9), seller(0.2)];
        assert_close(algorithm2(&order, 1, uniform_quantile), 0.0);
        assert_close(single_seller_opt(&order), 0.0);
    }

    #[test]
    fn price_equal_to_threshold_does_not_trade_except_force_sell() {
        let threshold = 0.5;
        // Equality does not buy.
        let order = [seller(0.5), buyer(0.9)];
        assert_close(algorithm1(&order, 1, threshold), 0.0);

        // Equality does not sell before the last buyer, then the last buyer force-sells.
        let order = [seller(0.2), buyer(0.5), buyer(0.1)];
        assert_close(algorithm1(&order, 2, threshold), 0.1 - 0.2);
    }

    #[test]
    fn alg2_sells_to_first_buyer_above_threshold_else_force_sells() {
        // n = 2, seller first, q = 2, T = 2/3.
        // First later buyer is below T; force-sell to position n+1.
        let order = [seller(0.1), buyer(0.2), buyer(0.3)];
        assert_close(algorithm2(&order, 2, uniform_quantile), 0.3 - 0.1);

        // First later buyer is above T, so stop there.
        let order = [seller(0.1), buyer(0.9), buyer(0.95)];
        assert_close(algorithm2(&order, 2, uniform_quantile), 0.9 - 0.1);

        // Seller in the middle: the early buyer is irrelevant; force-sell to the last agent.
        let order = [buyer(0.99), seller(0.1), buyer(0.2)];
        assert_close(algorithm2(&order, 2, uniform_quantile), 0.2 - 0.1);
        assert_close(single_seller_opt(&order), 0.2 - 0.1);
    }

    #[test]
    fn invariants_random_orders() {
        let mut rng = XorShift64::new(0xA5A5_1234_F00D);
        for n in 1..=6 {
            for m in 1..=4 {
                for _ in 0..40 {
                    let agents = random_market(&mut rng, n, m);
                    let threshold = rho(n, m);
                    let gft = algorithm1(&agents, n, threshold);
                    let checked = checked_algorithm1(&agents, n, threshold);
                    assert_close(gft, checked);
                }
            }
        }
    }

    #[test]
    fn monte_carlo_uniform_single_seller() {
        let n = 5usize;
        let m = 1usize;
        let trials = 20_000usize;
        let threshold = rho(n, m);
        assert_close(threshold, uniform_quantile(rho(n, m)));

        let mut rng = XorShift64::new(0xC0FF_EE00_D15E_A5E5);
        let mut sum_alg1 = 0.0;
        let mut sum_alg2 = 0.0;
        let mut sum_opt = 0.0;

        for _ in 0..trials {
            let agents = random_market(&mut rng, n, m);
            sum_alg1 += algorithm1(&agents, n, threshold);
            sum_alg2 += algorithm2(&agents, n, uniform_quantile);
            sum_opt += single_seller_opt(&agents);
        }

        let e_alg1 = sum_alg1 / trials as f64;
        let e_alg2 = sum_alg2 / trials as f64;
        let e_opt = sum_opt / trials as f64;
        assert!(e_alg1 > 0.0, "E[ALG1] = {e_alg1}");
        assert!(e_alg2 > 0.0, "E[ALG2] = {e_alg2}");

        let ratio1 = e_opt / e_alg1;
        let ratio2 = e_opt / e_alg2;
        assert!(
            ratio1 <= r(5) + 0.15,
            "E[OPT]/E[ALG1] = {ratio1}, R(5) + 0.15 = {}",
            r(5) + 0.15
        );
        assert!(
            ratio2 <= 2.0 + 0.15,
            "E[OPT]/E[ALG2] = {ratio2}"
        );
    }
}
