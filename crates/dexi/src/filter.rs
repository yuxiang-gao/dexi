//! Low-pass filter implementation.

/// First-order low-pass filter: y = y + alpha * (x - y)
#[derive(Debug, Clone)]
pub struct LPFilter {
    pub alpha: f64,
    pub y: Option<Vec<f64>>,
    pub is_init: bool,
}

impl LPFilter {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            y: None,
            is_init: false,
        }
    }

    pub fn next(&mut self, x: &[f64]) -> Vec<f64> {
        if !self.is_init {
            self.y = Some(x.to_vec());
            self.is_init = true;
            return self.y.as_ref().unwrap().clone();
        }
        let y = self.y.as_mut().unwrap();
        for i in 0..y.len() {
            y[i] = y[i] + self.alpha * (x[i] - y[i]);
        }
        y.clone()
    }

    pub fn reset(&mut self) {
        self.y = None;
        self.is_init = false;
    }
}
