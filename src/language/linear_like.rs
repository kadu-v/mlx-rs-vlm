use mlx::Array;
use mlx::builder::Builder;
use mlx::error::Exception;
use mlx::module::{Module, ModuleParam, ModuleParamMut, ModuleParamRef, ModuleParameters, Param};
use mlx_rs as mlx;

use crate::weights::{QuantizationMode, QuantizationSpec};

#[derive(Debug)]
pub enum LinearLike {
    Dense(mlx::nn::Linear),
    Quantized(mlx::nn::QuantizedLinear),
}

impl LinearLike {
    pub fn new(
        in_features: i32,
        out_features: i32,
        bias: bool,
        quant: Option<QuantizationSpec>,
    ) -> Result<Self, Exception> {
        match quant {
            Some(spec) => match spec.mode {
                QuantizationMode::Affine => Ok(Self::Quantized(
                    mlx::nn::QuantizedLinearBuilder::new(in_features, out_features)
                        .group_size(spec.group_size)
                        .bits(spec.bits)
                        .bias(bias)
                        .build()?,
                )),
            },
            None => {
                let mut l = mlx::nn::Linear::new(in_features, out_features)?;
                if !bias {
                    l.bias = Param::new(None);
                }
                Ok(Self::Dense(l))
            }
        }
    }
}

impl Module<&Array> for LinearLike {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        match self {
            Self::Dense(l) => l.training_mode(mode),
            Self::Quantized(q) => q.training_mode(mode),
        }
    }
}

impl ModuleParameters for LinearLike {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(l) => l.num_parameters(),
            Self::Quantized(q) => q.num_parameters(),
        }
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(l) => l.parameters(),
            Self::Quantized(q) => q.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Dense(l) => l.parameters_mut(),
            Self::Quantized(q) => q.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(l) => l.trainable_parameters(),
            Self::Quantized(q) => q.trainable_parameters(),
        }
    }

    fn update(&mut self, parameters: ModuleParam) {
        match self {
            Self::Dense(l) => l.update(parameters),
            Self::Quantized(q) => q.update(parameters),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(l) => l.freeze_parameters(recursive),
            Self::Quantized(q) => q.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(l) => l.unfreeze_parameters(recursive),
            Self::Quantized(q) => q.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(l) => l.all_frozen(),
            Self::Quantized(q) => q.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(l) => l.any_frozen(),
            Self::Quantized(q) => q.any_frozen(),
        }
    }
}

#[derive(Debug)]
pub enum EmbeddingLike {
    Dense(mlx::nn::Embedding),
    Quantized(mlx::nn::QuantizedEmbedding),
}

impl EmbeddingLike {
    pub fn new(
        num_embeddings: i32,
        dims: i32,
        quant: Option<QuantizationSpec>,
    ) -> Result<Self, Exception> {
        match quant {
            Some(spec) => match spec.mode {
                QuantizationMode::Affine => Ok(Self::Quantized(
                    mlx::nn::QuantizedEmbeddingBuilder::new(num_embeddings, dims)
                        .group_size(spec.group_size)
                        .bits(spec.bits)
                        .build()?,
                )),
            },
            None => Ok(Self::Dense(mlx::nn::Embedding::new(num_embeddings, dims)?)),
        }
    }

    pub fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Dense(e) => e.as_linear(x),
            Self::Quantized(q) => q.as_linear(x),
        }
    }
}

impl Module<&Array> for EmbeddingLike {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        match self {
            Self::Dense(e) => e.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        match self {
            Self::Dense(e) => e.training_mode(mode),
            Self::Quantized(q) => q.training_mode(mode),
        }
    }
}

impl ModuleParameters for EmbeddingLike {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(e) => e.num_parameters(),
            Self::Quantized(q) => q.num_parameters(),
        }
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(e) => e.parameters(),
            Self::Quantized(q) => q.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Dense(e) => e.parameters_mut(),
            Self::Quantized(q) => q.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(e) => e.trainable_parameters(),
            Self::Quantized(q) => q.trainable_parameters(),
        }
    }

    fn update(&mut self, parameters: ModuleParam) {
        match self {
            Self::Dense(e) => e.update(parameters),
            Self::Quantized(q) => q.update(parameters),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(e) => e.freeze_parameters(recursive),
            Self::Quantized(q) => q.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(e) => e.unfreeze_parameters(recursive),
            Self::Quantized(q) => q.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(e) => e.all_frozen(),
            Self::Quantized(q) => q.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(e) => e.any_frozen(),
            Self::Quantized(q) => q.any_frozen(),
        }
    }
}
