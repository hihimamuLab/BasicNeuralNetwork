use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::prelude::*;
use rand;
mod api;
use api::{activate_functions, reshape};

type NeuronLayer = Vec<f64>;
type DatasetVector = Vec<Vec<f32>>;
type WeightVector = Vec<Vec<f64>>;

const TRAIN_IMAGE: usize = 60000;
const TEST_IMAGE: usize = 10000;
const PIXEL: usize = 784;
const LABEL_LEN: usize = 10;

#[derive(Default)]
struct Weight {
    width: u16,
    height: u16,
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct Dataset {
    trn_img: DatasetVector,
    trn_lbl: DatasetVector,
    tst_img: DatasetVector,
    tst_lbl: DatasetVector,
}

#[derive(Debug)]
struct NetworkLayer;

enum ActivateType {
    Sigmoid,
    Softmax,
}

impl NetworkLayer {
    fn activate(neuron_layer: NeuronLayer, activate_type: ActivateType) -> NeuronLayer {
        match activate_type {
            ActivateType::Sigmoid => activate_functions::sigmoid(neuron_layer),
            ActivateType::Softmax => activate_functions::softmax(neuron_layer),
        }
    }
    fn forward_prop(
        neuron_layer: NeuronLayer,
        next_layer_len: usize,
        weights: WeightVector,
        bias: f64,
    ) -> NeuronLayer {
        let weights =
            Array::from_shape_vec((neuron_layer.len(), next_layer_len), weights.concat()).unwrap();
        let neuron_layer = Array::from_vec(neuron_layer);
        let bias = Array::from_vec(vec![bias]);
        (neuron_layer.dot(&weights) + bias).to_vec()
    }
}

impl Weight {
    fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
    fn size(&self, w: u16, h: u16) -> Self {
        Self {
            width: w,
            height: h,
        }
    }
    fn generate(&self) -> WeightVector {
        (0..self.height)
            .map(|_| (0..self.width).map(|_| rand::random::<f64>()).collect())
            .collect()
    }
}

impl Dataset {
    fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
    fn load_mnist(&self) -> Self {
        let mnist: NormalizedMnist = MnistBuilder::new()
            .label_format_one_hot()
            .base_path("./data")
            .finalize()
            .normalize();
        let trn_lbl: Vec<f32> = mnist.trn_lbl.into_iter().map(|v| v as f32).collect();
        let tst_lbl: Vec<f32> = mnist.tst_lbl.into_iter().map(|v| v as f32).collect();
        Self {
            trn_img: reshape(mnist.trn_img, TRAIN_IMAGE, PIXEL),
            trn_lbl: reshape(trn_lbl, TRAIN_IMAGE, LABEL_LEN),
            tst_img: reshape(mnist.tst_img, TEST_IMAGE, PIXEL),
            tst_lbl: reshape(tst_lbl, TEST_IMAGE, LABEL_LEN),
        }
    }
}

fn main() {
    let dataset: Dataset = Dataset::new().load_mnist();
    let layer_len_list: Vec<u16> = vec![100, 50, 10];
    let trn_img: NeuronLayer = dataset.trn_img[0].iter().map(|v| *v as f64).collect();
    let bias: f64 = 0.0;
    let weight_list: Vec<WeightVector> = vec![
        Weight::new()
            .size(layer_len_list[0], PIXEL.try_into().unwrap())
            .generate(),
        Weight::new()
            .size(layer_len_list[1], layer_len_list[0])
            .generate(),
        Weight::new()
            .size(layer_len_list[2], layer_len_list[1])
            .generate(),
    ];
}
