extern crate csv;
extern crate rulinalg;

use std::error::Error;
use std::io;
use std::process;
use std::str;
use std::f64::consts::E;
use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrixMut;
use rulinalg::matrix::BaseMatrix;

struct Example {
    label_value: u32,
    label : Matrix<f64>,
    values: Matrix<f64>,
}

fn sigmoid_loss(yhat : f64, y : f64) -> f64 {
    - (y*yhat.ln() + (1.0-y)*(1.0-yhat).ln())
}

fn loss(yhat: &Matrix<f64>, y: &Matrix<f64>) -> Matrix<f64> {
    let mut dat : Vec<f64> = Vec::new();
    for i in 0..yhat.rows() {
        let l = sigmoid_loss(yhat[[i, 0]], y[[i, 0]]);
        dat.push(l);
    }
    Matrix::new(10, 1, dat)
}

fn sigmoid(x : f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn print_dim(m : &Matrix<f64>) {
    println!("({},{})", m.rows(), m.cols());
}

fn compute_gradient(W: &Matrix<f64>, xi : &Matrix<f64>, b: &Matrix<f64>, y: &Matrix<f64>) -> Matrix<f64> {
    let z : Matrix<f64> = W.transpose() * xi + b;
    // activation.
    let a = z.apply(&sigmoid);
    // This transpose seems weird.
    xi * (a - y).transpose()
}


fn example() -> Result<(), Box<Error>> {
    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::Reader::from_reader(io::stdin());
    let mut examples : Vec<Example> = Vec::new();

    println!("Loading data...");
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        let record = result?;
        let label : u32 = record.get(0).unwrap().parse().unwrap();
        let mut dat = Vec::<f64>::with_capacity(10);
        for i in 0..10 {
            if i == label {
                dat.push(0.99);
            } else {
                dat.push(0.0);
            }
        }
        let mut label_matrix = Matrix::new(10, 1, dat);
        let mut values : Vec<f64> = Vec::new();
        for index in 0..(28*28) {
            let grayscale : u32 = record.get(1 + index).unwrap().parse().unwrap();
            values.push(grayscale as f64 / 256.0);
        }
        let mat = Matrix::new(28*28, 1, values);
        examples.push(Example {
            label_value: label,
            label: label_matrix,
            values: mat,
        });

    }
    println!("Training...");

    let batch_size = 20000;
    let learning_rate = 0.005;
    let mut W : Matrix<f64> = Matrix::zeros(28*28, 10);
    let mut b : Matrix<f64> = Matrix::zeros(10, 1);
    let mut dJ_dw : Matrix<f64> = Matrix::zeros(28*28, 10);
    let mut dJ_db : Matrix<f64> = Matrix::zeros(28*28, 10);

    for i in 0..batch_size {
        let xi = &examples.get(i).unwrap().values;
        let yhat = (W.transpose() * xi + &b).apply(&sigmoid);
        let y = &examples.get(i).unwrap().label;
        let loss = loss(&yhat, y);

//        println!("y {}", y);
//        println!("yhat {}", yhat);
//        println!("b {}", b);
//        println!("loss sum: {}", loss.sum());
//        println!("loss: {}", loss);

        let grad = compute_gradient(&W, xi, &b, y);
        W = W - grad * learning_rate;
    }

    let mut num_correct = 0;
    let total = 1000;
    for i in batch_size..batch_size+total {

        let example = examples.get(i).unwrap();

        let xi = &example.values;
        let yhat : Matrix<f64> = (W.transpose() * xi + &b).apply(&sigmoid);

        let mut label : u32 = 0;
        let mut highest_confidence = 0.0;
        for i in 0..10 {
            let confidence = yhat[[i, 0]];
            if confidence > highest_confidence {
                highest_confidence = confidence;
                label = i as u32;
            }
        }

        println!("guessed: {} got: {}", label, example.label_value);

        if label == example.label_value {
            num_correct+=1;
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