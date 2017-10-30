use std::f64::consts::E;

pub trait ActivationFunction {
    fn apply(x : f64) -> f64;
    fn derivative(x : f64) -> f64;
}

pub struct Sigmoid {}

impl ActivationFunction for Sigmoid {
    fn apply(x : f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    fn derivative(x : f64) -> f64  {
        let sig = Sigmoid::apply(x);
        sig * (1.0 - sig)
    }
}

pub struct TanH {}

impl ActivationFunction for TanH {
	fn apply(x : f64) -> f64  {
		x.tanh()
	}

	fn derivative(x : f64) -> f64  {
		let tanh = TanH::apply(x);
		1.0 - tanh * tanh
	}
}

pub struct ReLU {}

impl ActivationFunction for ReLU {
	fn apply(x : f64) -> f64  {
		if x > 0.0 {
			return x;
		}
		0.0
	}

	fn derivative(x : f64) -> f64  {
		if x > 0.0 {
			return 1.0;
		}
		0.0
	}
}