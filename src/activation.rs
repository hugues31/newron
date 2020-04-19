#[derive(Debug, Clone, Copy)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    TanH,
}

#[derive(Debug, Clone, Copy)]
pub struct Activation {
    activation: ActivationFunction,
}

impl Activation {
    pub fn relu() -> Activation {
        Activation {
            activation: ActivationFunction::ReLU,
        }
    }

    pub fn sigmoid() -> Activation {
        Activation {
            activation: ActivationFunction::Sigmoid,
        }
    }

    pub fn tanh() -> Activation {
        Activation {
            activation: ActivationFunction::TanH,
        }
    }

    pub fn activation(self) -> fn(f64) -> f64 {
        match self.activation {
            ActivationFunction::ReLU => relu_activation,
            ActivationFunction::Sigmoid => sigmoid_activation,
            ActivationFunction::TanH => tanh_activation,
        }
    }

    pub fn deriv_activation(self) -> fn(f64) -> f64 {
        match self.activation {
            ActivationFunction::ReLU => relu_deriv,
            ActivationFunction::Sigmoid => sigmoid_deriv,
            ActivationFunction::TanH => tanh_deriv,
        }
    }
}

pub fn relu_activation(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

pub fn relu_deriv(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        1.0
    }
}

pub fn sigmoid_activation(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

pub fn sigmoid_deriv(x: f64) -> f64 {
    sigmoid_activation(x) * (1. - sigmoid_activation(x))
}

pub fn tanh_activation(x: f64) -> f64 {
    (2.0 / (1.0 + (-2.0 * x).exp())) - 1.0
}

pub fn tanh_deriv(x: f64) -> f64 {
    1.0 - tanh_activation(x).powi(2)
}
