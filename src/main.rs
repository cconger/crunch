use anyhow::Result;
use clap::{App, Arg};
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use image::DynamicImage;
use tensorflow::{
    Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
};

#[derive(Debug)]
enum Label {
    Unlabeled,
    Gameplay,
    CharacterSelect,
    NotGameplay,
}

impl Label {
    fn from_u8(value: u8) -> Label {
        match value {
            3 => Label::Unlabeled,
            0 => Label::Gameplay,
            1 => Label::CharacterSelect,
            2 => Label::NotGameplay,
            _ => panic!("Unknown value: {}", value),
        }
    }
}

#[derive(Debug)]
struct ClassifierInput {
    image: DynamicImage,
}

impl ClassifierInput {
    fn from_path(path: &str) -> Result<Self> {
        let img = ImageReader::open(path)?
            .decode()?
            .resize_exact(480, 720, FilterType::Lanczos3);

        return Ok(Self { image: img });
    }

    fn to_tensor(&self) -> Result<Tensor<f32>> {
        let d = self
            .image
            .to_bytes()
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>();

        const INPUT_DIMS: &[u64] = &[1, 480, 720, 3];
        let tensor = Tensor::<f32>::new(INPUT_DIMS).with_values(&d)?;
        return Ok(tensor);
    }
}

#[derive(Debug)]
struct ClassifierOutput {
    label: Label,
    confidence: f32,
}

struct ClassifierModel {
    bundle: SavedModelBundle,
    input_op: Operation,
    input_index: i32,
    output_op: Operation,
    output_index: i32,
}

impl ClassifierModel {
    fn from_path(export_dir: &str) -> Result<Self> {
        const MODEL_TAG: &str = "serve";
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &[MODEL_TAG], &mut graph, export_dir)?;

        let sig = bundle
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
        let input_info = sig.get_input("input_2")?;
        let output_info = sig.get_output("dense")?;
        let input_op = graph.operation_by_name_required(&input_info.name().name)?;
        let output_op = graph.operation_by_name_required(&output_info.name().name)?;
        let input_index = input_info.name().index;
        let output_index = output_info.name().index;

        Ok(Self {
            bundle,
            input_op,
            input_index,
            output_op,
            output_index,
        })
    }

    fn predict(&self, image: ClassifierInput) -> Result<ClassifierOutput> {
        let input_tensor = image.to_tensor()?;
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&self.input_op, self.input_index, &input_tensor);
        let output_fetch = run_args.request_fetch(&self.output_op, self.output_index);
        self.bundle.session.run(&mut run_args)?;

        let output = run_args.fetch::<f32>(output_fetch)?;
        let mut confidence = 0f32;
        let mut label = 0u8;
        for i in 0..output.dims()[1] {
            println!("{}: {}", i, output[i as usize]);

            let conf = output[i as usize];
            if conf > confidence {
                confidence = conf;
                label = i as u8;
            }
        }

        Ok(ClassifierOutput {
            label: Label::from_u8(label),
            confidence,
        })
    }
}

fn main() {
    let matches = App::new("label")
        .version("0.0.1")
        .author("cconger@gmail.com")
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("input")
                .about("path to image to classify")
                .index(1)
                .required(true),
        )
        .get_matches();

    let model_path = matches.value_of("model").unwrap();
    let image_path = matches.value_of("input").unwrap();

    let model = ClassifierModel::from_path(model_path).unwrap();

    let input = ClassifierInput::from_path(image_path).unwrap();

    let result = model.predict(input).unwrap();
    println!("Got result: {:?}", result)
}
