extern crate csv;
extern crate rulinalg;
extern crate rand;

use std::error::Error;
use std::io;
use std::process;
use std::str;
use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrixMut;
use rulinalg::matrix::BaseMatrix;

use rand::{Rng, SeedableRng};

mod activations;
use activations::ActivationFunction;
use activations::Sigmoid;
use activations::TanH;
use activations::ReLU;

type Mat = Matrix<f64>;


pub fn reproducible_random_matrix(rows: usize, cols: usize) -> Matrix<f64> {
    const STANDARD_SEED: [usize; 4] = [12, 2049, 40, 33];
    let mut rng = rand::StdRng::from_seed(&STANDARD_SEED);
    let elements: Vec<_> = rng.gen_iter::<f64>()
        .take(rows * cols)
        .map(|n| (n - 0.5) / 100.0)
        .collect();
    Mat::new(rows, cols, elements)
}

struct Example {
    label_value: u32,
    label: Matrix<f64>,
    values: Matrix<f64>,
}

fn sigmoid_loss(yhat: f64, y: f64) -> f64 {
    -(y * yhat.ln() + (1.0 - y) * (1.0 - yhat).ln())
}

fn loss(yhat: &Matrix<f64>, y: &Matrix<f64>) -> Matrix<f64> {
    let mut dat: Vec<f64> = Vec::new();
    for i in 0..yhat.rows() {
        let l = sigmoid_loss(yhat[[i, 0]], y[[i, 0]]);
        dat.push(l);
    }
    Matrix::new(10, 1, dat)
}

fn print_dim(m: &Matrix<f64>) {
    println!("({},{})", m.rows(), m.cols());
}


fn example() -> Result<(), Box<Error>> {
    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::Reader::from_reader(io::stdin());
    let mut examples: Vec<Example> = Vec::new();

    println!("Loading data...");
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        let record = result?;
        let label: u32 = record.get(0).unwrap().parse().unwrap();
        let mut dat = Vec::<f64>::with_capacity(10);
        for i in 0..10 {
            if i == label {
                dat.push(0.99);
            } else {
                dat.push(0.0);
            }
        }
        let mut label_matrix = Matrix::new(10, 1, dat);
        let mut values: Vec<f64> = Vec::new();
        for index in 0..(28 * 28) {
            let grayscale: u32 = record.get(1 + index).unwrap().parse().unwrap();
            values.push(grayscale as f64 / 256.0);
        }
        let mat = Matrix::new(28 * 28, 1, values);
        examples.push(Example {
                          label_value: label,
                          label: label_matrix,
                          values: mat,
                      });

    }
    println!("Loaded {} examples.", examples.len());
    println!("Training...");

    let num_training_examples = 40000;
    let examples_per_batch = 1;
    let learning_rate = 0.01;
    let hidden_layer_nodes = 20;

    let mut w0: Matrix<f64> = reproducible_random_matrix(3 * 3, 1); // 3x3 convolution layer kernel
    let mut W1: Matrix<f64> = reproducible_random_matrix(28 * 28, hidden_layer_nodes); // 784x16
    let mut W2: Matrix<f64> = reproducible_random_matrix(hidden_layer_nodes, 10); // 16x10

    let mut b1: Matrix<f64> = Matrix::zeros(hidden_layer_nodes, 1);
    let mut b2: Matrix<f64> = Matrix::zeros(10, 1);

    // let mut dJ_dw1: Matrix<f64> = Matrix::zeros(28 * 28, 10);
    // let mut dJ_dw2: Matrix<f64> = Matrix::zeros(28 * 28, 10);

    for i in 0..num_training_examples / examples_per_batch {
        // Run in batches
        //println!("{}/{}", i, num_batches);
        let mut dw1 = Matrix::zeros(28 * 28, hidden_layer_nodes);
        let mut dw2 = Matrix::zeros(hidden_layer_nodes, 10);
        let mut db1 = Matrix::zeros(hidden_layer_nodes, 1);
        let mut db2 = Matrix::zeros(10, 1);
        for example in i * examples_per_batch..i * examples_per_batch + examples_per_batch {
            // Network structure X -> Conv(w[0]) -> a[0] -> W[1] -> a[1] -> W[2] -> a[2]
            // z[0] = conv(w[0], X)
            // a[0] = sigmoid(z[0])
            // z[1] = W[1](T)a[0] + b[1]
            // a[1] = tanh(z[1])
            // z[2] = W[2](T)a[1] + b[2]
            // a[2] = sigmoid(z[2])
            // Loss(a[2], y) = -a[2]*log(y) + (1 - y)log(1 - a[2])
            let xi = &examples.get(example).unwrap().values; // 784x1
            let z1 = W1.transpose() * xi + &b1;
            let a1 : Matrix<f64> = (z1.clone()).apply(&|x| ReLU::apply(x)); // 16x1
            let z2 = W2.transpose() * &a1 + &b2;
            let a2 = z2.apply(&|x| Sigmoid::apply(x)); // 10x1
            let y = &examples.get(i).unwrap().label; // 10x1
            // let loss = loss(&a2, y); // 10x1

            // dL/dw2 = dL/da2 * da2/dz2 * dz2/dW2
            let dLdW2 = &a1 * (&a2 - y).transpose(); // -> 16x10
            // dL/db2 = dL/da2 * da2/dz2 * dz2/db2
            //       = (a2 - a) * 1
            db2 = db2 + (&a2 - y);

            // dL/dw1 = dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dW1
            // Assume sigmoid in hidden layer: dsig/dx(x) = sig(x) * (1-sig(x))
            let dLdz2 = &a2 - y; // 10x1
            let dLda1 = &W2 * dLdz2; // 16x10 * 10x1 => 16x1
            let dLdz1 = dLda1.elemul(&z1.apply(&|x| ReLU::derivative(x))); // Apply sigmoid derivative => 16x1
            let dLdW1 = xi * dLdz1.transpose(); // 784x16
            // dL/db1 =  dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/db1
            db1 = db1 + dLdz1;

            // Update batch diff.
            dw1 = dw1 + dLdW1;
            dw2 = dw2 + dLdW2;
        }
        let rate = learning_rate / examples_per_batch as f64;
        W1 = W1 - dw1 * rate;
        W2 = W2 - dw2 * rate;
        b1 = b1 - db1 * rate;
        b2 = b2 - db2 * rate;
    }

    println!("Evaluate...");
    let mut num_correct = 0;
    let total = 1000;
    for i in num_training_examples..num_training_examples + total {

        let example = examples.get(i).unwrap();

        let xi = &example.values;
        let a1 = (W1.transpose() * xi + &b1).apply(&|x| ReLU::apply(x));
        let a2 = (W2.transpose() * a1 + &b2).apply(&|x| Sigmoid::apply(x)); // yhat

        let mut label: u32 = 0;
        let mut highest_confidence = 0.0;
        for i in 0..10 {
            let confidence = a2[[i, 0]];
            if confidence > highest_confidence {
                highest_confidence = confidence;
                label = i as u32;
            }
        }

        if label == example.label_value {
            num_correct += 1;
        }
    }

    println!("total correct / {}: {}", total, num_correct);
    println!("percent {}", num_correct as f32 / total as f32 * 100.0);

    Ok(())
}


fn main() {
    if let Err(err) = example() {
        println!("error running example: {}", err);
        process::exit(1);
    }
}
