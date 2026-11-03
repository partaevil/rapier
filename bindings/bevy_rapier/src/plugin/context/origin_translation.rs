//! Checked rigid-body/cache primitives for a caller-owned origin-shift transaction.
use super::RapierRigidBodySet;
use crate::dynamics::TransformInterpolation;
use crate::math::Vect;
use bevy::prelude::GlobalTransform;

/// Why an origin-translation primitive refused before changing its owned state.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OriginTranslationError {
    /// The requested translation contains NaN or infinity.
    NonFiniteTranslation,
    /// Soft bodies, proxies or their entity mappings need a separate transaction.
    SoftBodiesPresent,
    /// An owned world point is nonfinite or its translated value would overflow.
    NonFiniteState,
}

impl core::fmt::Display for OriginTranslationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NonFiniteTranslation => "origin translation must be finite",
            Self::SoftBodiesPresent => "origin translation requires a rigid-only body set",
            Self::NonFiniteState => "origin translation would leave nonfinite world coordinates",
        })
    }
}

impl std::error::Error for OriginTranslationError {}

fn validate_point(point: Vect, translation: Vect) -> Result<(), OriginTranslationError> {
    if !(point.is_finite() && (point + translation).is_finite()) {
        return Err(OriginTranslationError::NonFiniteState);
    }
    Ok(())
}

impl RapierRigidBodySet {
    /// Adds `translation` to native current/next poses and the last-written transform cache.
    ///
    /// Every owned point is checked before mutation. Soft bodies, remaining soft-body proxies
    /// and soft-body entity mappings reject, even for a zero translation. Transform bases
    /// (including shear/scale and the 2D render-order Z coordinate) are preserved exactly.
    ///
    /// This is one primitive, not a complete context transaction. Between SyncBackend and
    /// StepSimulation the caller must also translate colliders, multibody/contact state,
    /// query bounds, ECS poses, interpolation and other world-coordinate caches before
    /// exposing the new frame. Preflight those other owners before calling this method.
    pub fn try_translate_origin(
        &mut self,
        translation: Vect,
    ) -> Result<(), OriginTranslationError> {
        if !translation.is_finite() {
            return Err(OriginTranslationError::NonFiniteTranslation);
        }
        if !self.soft_bodies.is_empty()
            || !self.entity2soft_body.is_empty()
            || self
                .bodies
                .iter()
                .any(|(_, body)| body.soft_body().is_some())
        {
            return Err(OriginTranslationError::SoftBodiesPresent);
        }
        let shift = render_translation(translation);
        for (_, body) in self.bodies.iter() {
            validate_point(body.position().translation, translation)?;
            validate_point(body.next_position().translation, translation)?;
            validate_point(body.center_of_mass(), translation)?;
            let mut pose = *body.position();
            pose.translation += translation;
            if !pose
                .transform_point(body.local_center_of_mass())
                .is_finite()
            {
                return Err(OriginTranslationError::NonFiniteState);
            }
        }
        for transform in self.last_body_transform_set.values() {
            let affine = transform.affine();
            if !affine.matrix3.is_finite()
                || !affine.translation.is_finite()
                || !(affine.translation + shift).is_finite()
            {
                return Err(OriginTranslationError::NonFiniteState);
            }
        }
        self.bodies.translate_origin(translation);
        for transform in self.last_body_transform_set.values_mut() {
            let mut affine = transform.affine();
            affine.translation += shift;
            *transform = GlobalTransform::from(affine);
        }
        Ok(())
    }
}

fn render_translation(translation: Vect) -> bevy::math::Vec3A {
    #[cfg(feature = "dim2")]
    let translation = translation.extend(0.);
    translation.into()
}

impl TransformInterpolation {
    /// Adds a finite origin translation to both world-pose endpoints, preserving missing ones.
    ///
    /// Both endpoints are checked before either changes. The caller must preflight and commit
    /// this with the complete context/ECS origin transaction; this method shifts only this
    /// interpolation component and never advances or resets its interpolation history.
    pub fn try_translate_origin(
        &mut self,
        translation: Vect,
    ) -> Result<(), OriginTranslationError> {
        if !translation.is_finite() {
            return Err(OriginTranslationError::NonFiniteTranslation);
        }
        for pose in [self.start, self.end].into_iter().flatten() {
            validate_point(pose.translation, translation)?;
        }
        for pose in [&mut self.start, &mut self.end].into_iter().flatten() {
            pose.translation += translation;
        }
        Ok(())
    }
}
