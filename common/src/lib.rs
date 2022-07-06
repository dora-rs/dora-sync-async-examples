use dora_tracing::init_tracing;
use futures::StreamExt;
use image::{ImageBuffer, Rgb};
use onnxruntime::ndarray::ArrayBase;
use onnxruntime::{environment::Environment, session::Session};
use opentelemetry::{
    global,
    trace::{TraceContextExt, Tracer},
    Context,
};
use pollster::FutureExt as _;
use std::sync::{Arc, Mutex};
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
use tract_onnx::{
    model, prelude::Tensor, prelude::*, tract_hir::internal::tract_smallvec::SmallVec,
};

static MODEL_PATH: &str = "./data/squeezenet1.1-7.onnx";

pub fn load_model_tract(
) -> SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>> {
    tract_onnx::onnx()
        // load the model
        //.model_for_path("./data/efficientnet-lite4-11.onnx")
        .model_for_path(MODEL_PATH)
        .unwrap()
        // specify input type and shape
        //.with_input_fact(0, f32::fact(&[1i32, 3, 224, 224]).into())
        //.unwrap()
        // optimize the model
        .into_optimized()
        .unwrap()
        // make the model runnable and fix its inputs and outputs
        .into_runnable()
        .unwrap()
}

pub fn get_imagenet_labels() -> Vec<String> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data")
        .join("labels_map.txt");
    let file = BufReader::new(fs::File::open(labels_path).unwrap());

    file.lines().map(|line| line.unwrap()).collect()
}

pub fn preprocess(data: &[u8]) -> Tensor {
    let image: ImageBuffer<Rgb<u8>, &[u8]> =
        image::ImageBuffer::from_raw(1600, 1065, data).unwrap();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, x, y)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
    .into()
}

pub fn postprocess(results: SmallVec<[Arc<Tensor>; 4]>, class_labels: &[String]) -> String {
    let best = results[0]
        .to_array_view::<f32>()
        .unwrap()
        .iter()
        .cloned()
        .zip(2..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();
    //class_labels[best.1].clone()
    String::from("placeholder")
}

pub fn run(
    model: &SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    image: Tensor,
) -> TVec<Arc<Tensor>> {
    model.run(tvec!(image)).unwrap()
}

pub fn preprocess_gpu(
    data: &[u8],
) -> ArrayBase<onnxruntime::ndarray::OwnedRepr<f32>, onnxruntime::ndarray::Dim<[usize; 4]>> {
    let image: ImageBuffer<Rgb<u8>, &[u8]> =
        image::ImageBuffer::from_raw(1600, 1065, data).unwrap();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    ArrayBase::from_shape_fn((1, 3, 224, 224), |(_, c, x, y)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
    .into()
}

pub fn postprocess_gpu(
    results: Vec<
        onnxruntime::tensor::OrtOwnedTensor<
            f32,
            onnxruntime::ndarray::Dim<onnxruntime::ndarray::IxDynImpl>,
        >,
    >,
    // class_labels: &[String],
) -> String {
    let best = results[0]
        //   .to_array_view::<f32>()
        //   .unwrap()
        .iter()
        .cloned()
        .zip(2..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();
    //class_labels[best.1].clone()
    String::from("placeholder")
}

pub fn run_gpu(
    model: Arc<Mutex<Session>>,
    image: ArrayBase<onnxruntime::ndarray::OwnedRepr<f32>, onnxruntime::ndarray::Dim<[usize; 4]>>,
) -> String {
    let mut model = model.lock().unwrap();
    let result = model.run(vec![image]).unwrap();
    postprocess_gpu(result)
}

pub fn load_model_gpu() -> Session {
    let environment = Environment::builder()
        .with_name("integration_test")
        //.with_log_level(LoggingLevel::Warning)
        .build()
        .unwrap();

    let mut session = environment
        .new_session_builder()
        .unwrap()
        .use_cuda(0)
        .unwrap()
        .with_model_from_file(MODEL_PATH)
        .unwrap();

    let image = ArrayBase::from_shape_fn((1, 3, 224, 224), |(_, c, x, y)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (0 as f32 / 255.0 - mean) / std
    })
    .into();
    {
        let _results: Vec<
            onnxruntime::tensor::OrtOwnedTensor<
                f32,
                onnxruntime::ndarray::Dim<onnxruntime::ndarray::IxDynImpl>,
            >,
        > = session.run(vec![image]).unwrap();
    }
    session
}
