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
        let maximum_value: f64 = neuron_layer.iter().map(|v| *v).reduce(f64::max).unwrap();
        let sum_e_pow_element: f64 = neuron_layer
            .iter()
            .map(|v| Napier.powf(v - maximum_value))
            .sum();
        neuron_layer
            .iter()
            .map(|v| Napier.powf(v - maximum_value) / sum_e_pow_element)
            .collect()
    }
}

pub mod loss_function {
    use crate::{DataVector, NeuronLayer, LABEL_LEN};
    use std::f64::consts::E as Napier;

    pub fn cross_entropy_error(neuron_layer: Vec<NeuronLayer>, lbl: DataVector) -> f64 {
        let sum_loss: f64 = (0..neuron_layer.len())
            .map(|i| {
                (0..LABEL_LEN)
                    .map(|j| neuron_layer[i][j] * (lbl[i][j] as f64).log(Napier))
                    .sum::<f64>()
            })
            .sum::<f64>();

        -(sum_loss) / neuron_layer.len() as f64
    }
}

use crate::DataVector;
pub fn reshape(vector: Vec<f64>, width: usize) -> DataVector {
    let slice: &[f64] = vector.as_slice();
    slice.chunks(width).map(|v| v.to_vec()).collect()
}