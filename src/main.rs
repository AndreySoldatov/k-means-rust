use approx::RelativeEq;
use colors_transform::Color;
use image::DynamicImage;
use image::GenericImage;
use image::GenericImageView;
use image::ImageReader;
use image::Pixel;
use image::Rgb;
use image::RgbImage;
use nalgebra::Vector3;
use rand::prelude::*;

type NormImage = Vec<Vector3<f32>>;
type Cluster = Vec<Vector3<f32>>;
type Centroid = Vector3<f32>;

fn norm_vec3_from_rgb(col: Rgb<u8>) -> Vector3<f32> {
    Vector3::from_vec(col.0.into_iter().map(|x| x as f32 / 255.0 ).collect::<Vec<f32>>())
}

fn norm_vec3_to_rgb(vec: Vector3<f32>) -> Rgb<u8> {
    Rgb::from([
        (vec.x * 255.0) as u8,
        (vec.y * 255.0) as u8,
        (vec.z * 255.0) as u8
    ])
}

fn dyn_image_to_norm_vec(img: DynamicImage) -> NormImage {
    let mut res: NormImage = Vec::with_capacity(img.height() as usize);

    for (_, _, p) in img.pixels() {
        res.push(norm_vec3_from_rgb(p.to_rgb()))
    }

    res
}

fn calc_new_centroid(cluster: &Cluster) -> Centroid {
    let mut accum: Centroid = Vector3::zeros();

    for p in cluster {
        accum += p;
    }

    accum / (cluster.len() as f32)
}

fn k_cluster_rgb_image(k: usize, img: &NormImage, eps: f32) -> Vec<Centroid> {
    let mut rng = thread_rng();

    let mut centroids: Vec<Centroid> =  vec![];
    for _ in 0..k {
        centroids.push(Vector3::new(
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
        ));
    }

    let mut conv = false;

    while !conv {
        let mut clusters: Vec<Cluster> = vec![vec![]; k];

        for p in img {
            let min = centroids
                .clone()
                .into_iter()
                .map(|v| p.metric_distance(&v) )
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap();
            clusters[min].push(p.clone());
        }

        let new_centroids: Vec<Centroid> = clusters.iter().map(|c| {
            if c.len() > 0 {
                calc_new_centroid(c)
            } else {
                Vector3::new(
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                )
            }
        }).collect();

        let mut equal = true;
        for (c1, c2) in new_centroids.iter().zip(centroids.iter()) {
            equal &= c1.relative_eq(c2, eps, f32::default_max_relative());
        }
        conv = equal;
        
        centroids = new_centroids;
    }

    centroids
}

fn main() {
    let (width, height) = (256, 256);
    let img = ImageReader::open(std::env::args().nth(1).unwrap()).unwrap().decode().unwrap().thumbnail(width, height);
    let img = dyn_image_to_norm_vec(img);

    let k: usize = std::env::args().nth(2).unwrap_or(String::from("6")).parse().unwrap_or(6);

    let mut res_image = DynamicImage::ImageRgb8(RgbImage::new(k as u32, 1));

    let mut palette = k_cluster_rgb_image(k, &img, 0.01)
        .into_iter()
        .map(|v| {
            colors_transform::Rgb::from(v.x * 255.0, v.y * 255.0, v.z * 255.0)
        }).collect::<Vec<colors_transform::Rgb>>();

    palette.sort_by(|a, b| a.get_hue().total_cmp(&b.get_hue()) );

    for (i, c) in palette.iter().enumerate() {
        res_image.put_pixel(i as u32, 0, image::Rgba([
            c.get_red() as u8,
            c.get_green() as u8,
            c.get_blue() as u8,
            255
        ]));
    }

    res_image = res_image.resize((k * 100) as u32, 100, image::imageops::FilterType::Nearest);
    res_image.save("res.png").unwrap();
}
