use image::{ImageBuffer, Rgb};
#[cfg(feature = "gpu")]
use onnxruntime::ndarray::ArrayBase;
#[cfg(feature = "gpu")]
use onnxruntime::{environment::Environment, session::Session};
use opentelemetry::sdk::trace::Span;
use opentelemetry::{
    global,
    trace::{TraceContextExt, Tracer},
    Context,
};
use std::sync::Arc;
#[cfg(feature = "gpu")]
use std::sync::Mutex;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
use tract_onnx::{prelude::Tensor, prelude::*, tract_hir::internal::tract_smallvec::SmallVec};

static MODEL_PATH: &str = "./data/efficientnet-lite4-11.onnx";

type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
pub fn load_model_tract() -> TractModel {
    tract_onnx::onnx()
        .model_for_path(MODEL_PATH)
        .unwrap()
        .into_optimized()
        .unwrap()
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

pub fn postprocess(results: SmallVec<[Arc<Tensor>; 4]>) -> (f32, i32) {
    let best = results[0]
        .to_array_view::<f32>()
        .unwrap()
        .iter()
        .cloned()
        .zip(2..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();
    best
}

pub fn run(model: &TractModel, data: &[u8], span: Span) -> (f32, i32) {
    let context = Context::current_with_span(span);
    let tracer = global::tracer("name");
    let _span = tracer.start_with_context("in_sync_thread", &context);
    let image = preprocess(data);
    postprocess(model.run(tvec!(image)).unwrap())
}

#[cfg(feature = "gpu")]
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
}

#[cfg(feature = "gpu")]
pub fn postprocess_gpu(
    results: Vec<
        onnxruntime::tensor::OrtOwnedTensor<
            f32,
            onnxruntime::ndarray::Dim<onnxruntime::ndarray::IxDynImpl>,
        >,
    >,
) -> (f32, i32) {
    let best = results[0]
        //   .to_array_view::<f32>()
        //   .unwrap()
        .iter()
        .cloned()
        .zip(2..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();
    best
}

#[cfg(feature = "gpu")]
pub fn run_gpu(
    model: Arc<Mutex<Session>>,
    image: ArrayBase<onnxruntime::ndarray::OwnedRepr<f32>, onnxruntime::ndarray::Dim<[usize; 4]>>,
) -> (f32, i32) {
    let mut model = model.lock().unwrap();
    let result = model.run(vec![image]).unwrap();
    postprocess_gpu(result)
}

#[cfg(feature = "gpu")]
pub fn load_model_gpu() -> Session {
    let environment = Environment::builder()
        .with_name("integration_test")
        .build()
        .unwrap();

    let mut session = environment
        .new_session_builder()
        .unwrap()
        .use_cuda(0)
        .unwrap()
        .with_model_from_file(MODEL_PATH)
        .unwrap();

    let image = ArrayBase::from_shape_fn((1, 3, 224, 224), |(_, _, _, _)| 0 as f32);
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
