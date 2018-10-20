#![cfg(feature = "arbitrary")]

use std::cmp;

use na::balancing;
use na::{DMatrix, Matrix4};

quickcheck! {
    fn balancing_parlett_reinsch(n: usize) -> bool {
        let n = cmp::min(n, 10);
        let m = DMatrix::<f64>::new_random(n, n);
        let mut balanced = m.clone();
        let d = balancing::balance_parlett_reinsch(&mut balanced);
        balancing::unbalance(&mut balanced, &d);

        balanced == m
    }

    fn balancing_parlett_reinsch_static(m: Matrix4<f64>) -> bool {
        let mut balanced = m;
        let d = balancing::balance_parlett_reinsch(&mut balanced);
        balancing::unbalance(&mut balanced, &d);

        balanced == m
    }
}
