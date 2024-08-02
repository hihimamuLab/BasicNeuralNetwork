use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::{prelude::*, Data};
use rand::Rng;
mod api;
use api::{activate_functions, reshape};

type NeuronLayer = Vec<f64>;
type DataVector = Vec<Vec<f64>>;

const TRAIN_IMAGE: usize = 60000;
const TEST_IMAGE: usize = 10000;
const PIXEL: usize = 784;
const LABEL_LEN: usize = 10;

#[derive(Default)]
struct Weight {
    width: u16,
    height: u16,
}

#[derive(Default)]
struct Bias;

#[derive(Default)]
struct Store {
    weights: Vec<DataVector>,
    biases: DataVector,
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct Dataset {
    trn_img: DataVector,
    trn_lbl: DataVector,
    tst_img: DataVector,
    tst_lbl: DataVector,
}

#[derive(Debug)]
struct Network;

enum ActivateType {
    Sigmoid,
    Softmax,
}

impl Network {
    fn activate(neuron_layer: NeuronLayer, activate_type: ActivateType) -> NeuronLayer {
        match activate_type {
            ActivateType::Sigmoid => activate_functions::sigmoid(neuron_layer),
            ActivateType::Softmax => activate_functions::softmax(neuron_layer),
        }
    }
    fn forward_prop(input: Vec<NeuronLayer>, weight: Vec<DataVector>, bias: DataVector) -> DataVector {
        let mut output: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> =
            Array::from_shape_vec((input.len(), input[0].len()), input.concat()).unwrap();
        let mut softmax_output: DataVector = Vec::new();
        for i in 0..weight.len() {
            let w: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> =
                Array::from_shape_vec((weight[i].len(), weight[i][0].len()), weight[i].concat())
                    .unwrap();
            let b: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = Array::from_vec(bias[i].clone());
            output = output.dot(&w) + b;
            if i < weight.len() - 1 {
                let output_vec: Vec<f64> = Array::into_raw_vec(output.clone());
                let reshaped_output_vec: DataVector = api::reshape(output_vec, weight[i][0].len());
                let sigmoid: DataVector = reshaped_output_vec.into_iter().map(|vec| Self::activate(vec, ActivateType::Sigmoid)).collect();
                output = Array::from_shape_vec((input.len(), weight[i][0].len()), sigmoid.concat()).unwrap();
            } else {
                let output_vec: Vec<f64> = Array::into_raw_vec(output.clone());
                let reshaped_output_vec: DataVector = api::reshape(output_vec, weight[i][0].len());
                softmax_output = reshaped_output_vec.into_iter().map(|vec| Self::activate(vec, ActivateType::Softmax)).collect();
            }
        }
        softmax_output
    }
    fn accuracy(output: Vec<NeuronLayer>, lbl: DataVector) -> f64 {
        let mut correct_count: i32 = 0;
        for i in 0..output.len() {
            let idx_max_output: usize = output[i]
                .iter()
                .position(|v| *v == output[i].clone().into_iter().reduce(f64::max).unwrap())
                .unwrap();
            let idx_max_lbl: usize = lbl[i]
                .iter()
                .position(|v| *v == lbl[i].clone().into_iter().reduce(f64::max).unwrap())
                .unwrap();
            if idx_max_output == idx_max_lbl {
                correct_count += 1;
            }
        }
        let accuracy: f64 = (correct_count / 100).into();
        accuracy
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
    fn generate(&self) -> DataVector {
        (0..self.height)
            .map(|_| {
                (0..self.width)
                    .map(|_| rand::thread_rng().gen_range(0.0..=0.1))
                    .collect()
            })
            .collect()
    }
    fn build(layer_len_list: Vec<u16>, input_len: u16) -> Vec<DataVector> {
        let mut weight_list: Vec<DataVector> = vec![];
        for i in 0..layer_len_list.len() {
            if i < 1 {
                weight_list.push(Self::new().size(layer_len_list[i], input_len).generate());
            } else {
                weight_list.push(
                    Self::new()
                        .size(layer_len_list[i], layer_len_list[i - 1])
                        .generate(),
                );
            }
        }
        weight_list
    }
}

impl Bias {
    fn build(network: Vec<u16>) -> DataVector {
        network.into_iter().map(|v| vec![0.0; v.into()]).collect()
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
        let trn_img: Vec<f64> = mnist.trn_img.iter().map(|v| *v as f64).collect();
        let trn_lbl: Vec<f64> = mnist.trn_lbl.iter().map(|v| *v as f64).collect();
        let tst_img: Vec<f64> = mnist.tst_img.iter().map(|v| *v as f64).collect();
        let tst_lbl: Vec<f64> = mnist.tst_lbl.iter().map(|v| *v as f64).collect();
        Self {
            trn_img: reshape(trn_img, PIXEL),
            trn_lbl: reshape(trn_lbl, LABEL_LEN),
            tst_img: reshape(tst_img, PIXEL),
            tst_lbl: reshape(tst_lbl, LABEL_LEN),
        }
    }
}

impl Store {
    fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
    fn weight(&mut self, weight: Vec<DataVector>) {
        self.weights = weight;
    }
    fn bias(&mut self, bias: DataVector) {
        self.biases = bias;
    }
}

fn main() {
    let dataset: Dataset = Dataset::new().load_mnist();
    let network: Vec<u16> = vec![784, 100, 50, 10];
    let weight: Vec<DataVector> = Weight::build(network[1..].to_vec(), network[0]);
    let bias: DataVector = Bias::build(network[1..].to_vec());
    let trn_img: Vec<NeuronLayer> = dataset.trn_img[0..4].to_vec();
    let trn_lbl: Vec<NeuronLayer> = dataset.trn_lbl[0..4].to_vec();
    let output: DataVector = Network::forward_prop(trn_img, weight, bias);
    println!("{}", api::loss_function::cross_entropy_error(output.clone(), trn_lbl.clone()));
}