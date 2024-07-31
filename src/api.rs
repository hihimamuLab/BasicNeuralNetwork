pub mod activate_functions {
    use crate::NeuronLayer;
    use std::f64::consts::E as Napier;

    pub fn sigmoid(neuron_layer: NeuronLayer) -> NeuronLayer {
        neuron_layer
            .into_iter()
            .map(|v| {
                let v: f64 = 1.0 / (1.0 + Napier.powf(v));
                v
            })
            .collect()
    }

    pub fn softmax(neuron_layer: NeuronLayer) -> NeuronLayer {
        let maximum_value: f64 = neuron_layer
            .iter()
            .map(|v| *v)
            .reduce(f64::max)
            .unwrap();
        let sum_e_pow_element: f64 = neuron_layer
            .iter()
            .map(|v| Napier.powf(v - maximum_value))
            .sum();
        neuron_layer
            .iter()
            .map(|v| {
                Napier.powf(v - maximum_value) / sum_e_pow_element
            })
            .collect()
    }
}

use crate::{DatasetVector, NeuronLayer};
pub fn reshape(mut vector: Vec<f32>, height: usize, width: usize) -> DatasetVector {
    let mut _cut_vector: Vec<f32> = vec![];
    let mut reshaped_vector: DatasetVector = vec![];
    for _ in 0..height {
        _cut_vector = vector.split_off(width);
        reshaped_vector.push(vector.clone());
    }
    reshaped_vector
}

pub fn max(neuron_layer: NeuronLayer) -> Vec<u8> {
    let max: f64 = neuron_layer.iter().map(|v| *v).reduce(f64::max).unwrap();
    let max_index: usize = neuron_layer.iter().position(|v| v == &max).unwrap();
    let mut lbl: Vec<u8> = vec![0; 10];
    lbl[max_index] = 1;
    lbl
}