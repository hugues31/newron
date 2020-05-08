use crate::layers::layer::Layer;

pub trait OptimizerStep {
    fn step(&self, layers: &mut [Box<dyn Layer>]);
}
