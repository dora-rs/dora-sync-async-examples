use dora_node_api::{self, DoraNode};
use dora_tracing::init_tracing;
use futures::StreamExt;
use image::{ImageBuffer, Rgb};
use onnxruntime::environment::Environment;
use opentelemetry::{
    global,
    trace::{Span, TraceContextExt, Tracer},
    Context,
};
use pollster::FutureExt as _;
use std::sync::Mutex;
use std::time::Duration;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
use tokio::sync::mpsc;
use tokio::time::sleep;
use tract_onnx::prelude::*;
fn main() -> eyre::Result<()> {
    let tracer = init_tracing("receiver").unwrap();
    //let model = tract_onnx::onnx()
    //// load the model
    //.model_for_path("./data/efficientnet-lite4-11.onnx")
    //.unwrap()
    //// specify input type and shape
    //.with_input_fact(0, f32::fact(&[1i32, 224, 224, 3]).into())
    //.unwrap()
    //// optimize the model
    //.into_optimized()
    //.unwrap()
    //// make the model runnable and fix its inputs and outputs
    //.into_runnable()
    //.unwrap();

    // let mut handles = Vec::with_capacity(40);
    let class_labels = get_imagenet_labels();

    let environment = Environment::builder()
        .with_name("integration_test")
        //.with_log_level(LoggingLevel::Warning)
        .build()
        .unwrap();

    let session = Arc::new(Mutex::new(
        environment
            .new_session_builder()
            .unwrap()
            .use_cuda(0)
            .unwrap()
            .with_model_from_file("./data/efficientnet-lite4-11.onnx")
            .unwrap(),
    ));
    let image = onnxruntime::ndarray::Array::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (0 as f32 / 255.0 - mean) / std
    })
    .into();
    // run the model on the input
    let input_tensor_values = vec![image];
    //let result = model.run(tvec!(image)).unwrap();
    {
        let mut _session = session.lock().unwrap();
        let result: Vec<
            onnxruntime::tensor::OrtOwnedTensor<
                f32,
                onnxruntime::ndarray::Dim<onnxruntime::ndarray::IxDynImpl>,
            >,
        > = _session.run(input_tensor_values).unwrap();
    }
    rayon::scope(|s| {
        let span = tracer.start("spawn-blocking");
        let context = Context::current_with_span(span);
        let node = DoraNode::init_from_env().block_on()?;
        // let class_labels = &class_labels;
        // let model = &model;

        let mut inputs = node.inputs().block_on()?;
        for call_id in 0..100 {
            let _span = tracer.start_with_context(format!("call-{call_id}"), &context);
            let input = match inputs.next().block_on() {
                Some(input) => input,
                None => {
                    println!("None");
                    continue;
                }
                _ => eyre::bail!(""),
            };
            let a = call_id.clone();
            let session = session.clone();
            //        let (send, recv) = tokio::sync::oneshot::channel();
            s.spawn(move |_| {
                let mut session = session.lock().unwrap();
                let _context = Context::current_with_span(_span);
                let tracer = global::tracer("name");
                let __span = tracer.start_with_context("tokio-spawn-blocking", &_context);
                let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
                    image::ImageBuffer::from_vec(1600, 1065, input.data).unwrap();
                let resized = image::imageops::resize(
                    &image,
                    224,
                    224,
                    ::image::imageops::FilterType::Triangle,
                );
                let image =
                    onnxruntime::ndarray::Array::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
                        let mean = [0.485, 0.456, 0.406][c];
                        let std = [0.229, 0.224, 0.225][c];
                        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
                    })
                    .into();
                // run the model on the input
                let input_tensor_values = vec![image];
                //let result = model.run(tvec!(image)).unwrap();
                let result: Vec<
                    onnxruntime::tensor::OrtOwnedTensor<
                        f32,
                        onnxruntime::ndarray::Dim<onnxruntime::ndarray::IxDynImpl>,
                    >,
                > = session.run(input_tensor_values).unwrap();
                // find and display the max value with its index
                let _best = result[0]
                    //    .to_array_view::<f32>()
                    //    .unwrap()
                    .iter()
                    .cloned()
                    .zip(2..)
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unwrap();

                // println!("result {a}: {:?}", class_labels[best.1]);
                //   _test.save(format!("outputs/output-{a}.jpg")).unwrap();
                //  tracer.add_event("test: {call_id}", vec![]);
                // send.send("").unwrap();
            });
            //     handles.push(recv);
        }
        Ok(())
    })
    .unwrap();
    // futures::future::join_all(handles).await;
    Ok(())
}

fn get_imagenet_labels() -> Vec<String> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data")
        .join("imagenet_slim_labels.txt");
    let file = BufReader::new(fs::File::open(labels_path).unwrap());

    file.lines().map(|line| line.unwrap()).collect()
}
