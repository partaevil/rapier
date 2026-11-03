//! Checked adapter and real SyncBackend/StepSimulation regression witnesses.
use super::{OriginTranslationError, RapierRigidBodySet};
use crate::math::Vect;
use crate::prelude::*;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy::time::{TimePlugin, TimeUpdateStrategy};
use rapier::math::Pose;
use rapier::prelude::{RigidBodyBuilder, SoftBodyBuilder, SoftBodySet};

fn v(x: f32, y: f32, z: f32) -> Vect {
    #[cfg(feature = "dim2")]
    {
        let _ = z;
        Vect::new(x, y)
    }
    #[cfg(feature = "dim3")]
    {
        Vect::new(x, y, z)
    }
}
fn render_delta(delta: Vect) -> Vec3 {
    #[cfg(feature = "dim2")]
    {
        delta.extend(0.)
    }
    #[cfg(feature = "dim3")]
    {
        delta
    }
}
fn snapshot(set: &RapierRigidBodySet) -> String {
    format!(
        "{:?}|{:?}|{:?}|{:?}|{}",
        set.bodies.iter().collect::<Vec<_>>(),
        set.last_body_transform_set,
        set.entity2body,
        set.entity2soft_body,
        set.soft_bodies.len()
    )
}
fn populated() -> (RapierRigidBodySet, rapier::dynamics::RigidBodyHandle) {
    let mut set = RapierRigidBodySet::default();
    let handle = set
        .bodies
        .insert(RigidBodyBuilder::kinematic_position_based().translation(v(1., 2., 3.)));
    set.bodies[handle].set_next_kinematic_translation(v(2., 3., 4.));
    let affine =
        GlobalTransform::from(Transform::from_xyz(3., 4., 17.).with_scale(Vec3::new(2., 3., 4.)))
            .mul_transform(Transform::from_rotation(Quat::from_rotation_z(0.4)));
    set.last_body_transform_set.insert(handle, affine);
    (set, handle)
}

#[test]
fn origin_translation_preserves_affine_cache_native_motion_and_2d_render_order() {
    let (mut set, handle) = populated();
    let before = set.last_body_transform_set[&handle].affine();
    let pose = *set.bodies[handle].position();
    let next = *set.bodies[handle].next_position();
    let delta = v(32., -16., 8.);
    set.try_translate_origin(delta).unwrap();
    let after = set.last_body_transform_set[&handle].affine();
    assert_eq!(after.matrix3, before.matrix3);
    assert_eq!(
        after.translation,
        before.translation + bevy::math::Vec3A::from(render_delta(delta))
    );
    assert_eq!(
        set.bodies[handle].position().translation,
        pose.translation + delta
    );
    assert_eq!(
        set.bodies[handle].next_position().translation,
        next.translation + delta
    );
    assert_eq!(set.bodies[handle].position().rotation, pose.rotation);
    set.try_translate_origin(-delta).unwrap();
    assert_eq!(set.last_body_transform_set[&handle].affine(), before);
}

#[test]
fn origin_translation_refuses_nonfinite_requests_without_owned_state_changes() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for mut set in [RapierRigidBodySet::default(), populated().0] {
            let before = snapshot(&set);
            assert_eq!(
                set.try_translate_origin(v(bad, 0., 0.)),
                Err(OriginTranslationError::NonFiniteTranslation)
            );
            assert_eq!(snapshot(&set), before);
        }
    }
}

#[test]
fn origin_translation_refuses_soft_bodies_stale_maps_and_orphan_proxies() {
    let (mut set, _) = populated();
    let mut colliders = rapier::geometry::ColliderSet::new();
    let soft = set.soft_bodies.insert(
        SoftBodyBuilder::rope(v(0., 2., 0.), v(1., 2., 0.), 3),
        &mut set.bodies,
        &mut colliders,
    );
    let before = snapshot(&set);
    assert_eq!(
        set.try_translate_origin(Vect::ZERO),
        Err(OriginTranslationError::SoftBodiesPresent)
    );
    assert_eq!(snapshot(&set), before);
    set.soft_bodies = SoftBodySet::default();
    let orphan = snapshot(&set);
    assert_eq!(
        set.try_translate_origin(v(1., 0., 0.)),
        Err(OriginTranslationError::SoftBodiesPresent)
    );
    assert_eq!(snapshot(&set), orphan);
    let mut stale = RapierRigidBodySet::default();
    stale.entity2soft_body.insert(Entity::from_bits(1), soft);
    let before = snapshot(&stale);
    assert_eq!(
        stale.try_translate_origin(Vect::ZERO),
        Err(OriginTranslationError::SoftBodiesPresent)
    );
    assert_eq!(snapshot(&stale), before);
}

#[test]
fn origin_translation_refuses_pose_target_or_cache_overflow_before_any_mutation() {
    for owner in 0..3 {
        let (mut set, handle) = populated();
        match owner {
            0 => set.bodies[handle].set_translation(v(f32::MAX, 0., 0.), false),
            1 => set.bodies[handle].set_next_kinematic_translation(v(f32::MAX, 0., 0.)),
            _ => {
                set.last_body_transform_set.insert(
                    handle,
                    GlobalTransform::from_translation(Vec3::new(f32::MAX, 0., 0.)),
                );
            }
        }
        let before = snapshot(&set);
        assert_eq!(
            set.try_translate_origin(v(f32::MAX, 0., 0.)),
            Err(OriginTranslationError::NonFiniteState)
        );
        assert_eq!(snapshot(&set), before, "owner {owner}");
    }
}

#[test]
fn origin_translation_interpolation_preserves_missing_endpoints_and_lerped_motion() {
    for endpoints in [(false, false), (true, false), (false, true), (true, true)] {
        let mut interpolation = TransformInterpolation {
            start: endpoints.0.then(|| Pose::from_translation(v(1., 2., 3.))),
            end: endpoints.1.then(|| Pose::from_translation(v(2., 3., 4.))),
        };
        let before = interpolation;
        let old_sample = before.lerp_slerp(0.25);
        let delta = v(16., -8., 4.);
        interpolation.try_translate_origin(delta).unwrap();
        for (a, b) in [
            before.start.zip(interpolation.start),
            before.end.zip(interpolation.end),
        ]
        .into_iter()
        .flatten()
        {
            assert_eq!(b.translation, a.translation + delta);
            assert_eq!(b.rotation, a.rotation);
        }
        assert_eq!(interpolation.start.is_some(), before.start.is_some());
        assert_eq!(interpolation.end.is_some(), before.end.is_some());
        if let Some((a, b)) = old_sample.zip(interpolation.lerp_slerp(0.25)) {
            assert_eq!(b.translation, a.translation + delta);
        }
        interpolation.try_translate_origin(-delta).unwrap();
        assert_eq!(interpolation, before);
    }
}

#[test]
fn origin_translation_interpolation_refuses_invalid_later_endpoint_atomically() {
    let mut interpolation = TransformInterpolation {
        start: Some(Pose::from_translation(v(1., 0., 0.))),
        end: Some(Pose::from_translation(v(f32::MAX, 0., 0.))),
    };
    let before = interpolation;
    assert_eq!(
        interpolation.try_translate_origin(v(f32::MAX, 0., 0.)),
        Err(OriginTranslationError::NonFiniteState)
    );
    assert_eq!(interpolation, before);
    let mut empty = TransformInterpolation::default();
    assert_eq!(
        empty.try_translate_origin(v(f32::NAN, 0., 0.)),
        Err(OriginTranslationError::NonFiniteTranslation)
    );
    assert_eq!(empty, TransformInterpolation::default());
}

#[derive(Component)]
struct Actor;
#[derive(Resource, Default)]
struct Request(Option<Vect>);
#[derive(Resource, Default)]
struct Outcome(Option<Result<(), OriginTranslationError>>);

// This fixture owns only root ECS poses. It preflights interpolation copies before committing
// the body/cache primitive and its native collider/joint/contact/query/ECS partners together.
fn apply_request(
    mut request: ResMut<Request>,
    mut outcome: ResMut<Outcome>,
    mut contexts: Query<(
        &mut RapierRigidBodySet,
        &mut RapierContextColliders,
        &mut RapierContextJoints,
        &mut RapierContextSimulation,
    )>,
    mut actors: Query<
        (
            &mut Transform,
            &mut GlobalTransform,
            Option<&mut TransformInterpolation>,
        ),
        With<Actor>,
    >,
) {
    let Some(delta) = request.0.take() else {
        return;
    };
    for (_, _, interpolation) in &actors {
        if let Some(interpolation) = interpolation {
            let mut candidate = *interpolation;
            if let Err(error) = candidate.try_translate_origin(delta) {
                outcome.0 = Some(Err(error));
                return;
            }
        }
    }
    let (mut bodies, mut colliders, mut joints, mut simulation) = contexts.single_mut().unwrap();
    if let Err(error) = bodies.try_translate_origin(delta) {
        outcome.0 = Some(Err(error));
        return;
    }
    colliders.colliders.translate_origin(delta);
    joints.multibody_joints.translate_origin(delta);
    simulation.narrow_phase.translate_origin(delta);
    let params = simulation.integration_parameters;
    for (handle, collider) in colliders.colliders.iter() {
        if collider.is_enabled() {
            simulation
                .broad_phase
                .set_aabb(&params, handle, collider.compute_aabb());
        }
    }
    for (mut transform, mut global, interpolation) in &mut actors {
        transform.translation += render_delta(delta);
        let mut affine = global.affine();
        affine.translation += bevy::math::Vec3A::from(render_delta(delta));
        *global = GlobalTransform::from(affine);
        if let Some(mut interpolation) = interpolation {
            interpolation.try_translate_origin(delta).unwrap();
        }
    }
    outcome.0 = Some(Ok(()));
}
fn app(interpolated: bool) -> App {
    let mut app = App::new();
    app.add_plugins((
        TransformPlugin,
        TimePlugin,
        RapierPhysicsPlugin::<NoUserData>::default(),
    ))
    .insert_resource(TimeUpdateStrategy::ManualDuration(
        std::time::Duration::from_secs_f32(1. / 60.),
    ))
    .insert_resource(if interpolated {
        TimestepMode::Interpolated {
            dt: 1. / 60.,
            time_scale: 1.,
            substeps: 1,
        }
    } else {
        TimestepMode::Fixed {
            dt: 1. / 60.,
            substeps: 1,
        }
    })
    .init_resource::<Request>()
    .init_resource::<Outcome>()
    .add_systems(
        PostUpdate,
        apply_request
            .after(PhysicsSet::SyncBackend)
            .before(PhysicsSet::StepSimulation),
    );
    app.finish();
    app.update();
    app
}
fn body_state(app: &mut App, entity: Entity) -> (Vect, Vect, Vect, bool) {
    let world = app.world_mut();
    let set = world.query::<&RapierRigidBodySet>().single(world).unwrap();
    let body = &set.bodies[set.entity2body[&entity]];
    (
        body.translation(),
        body.next_position().translation,
        body.linvel(),
        body.is_sleeping(),
    )
}

#[test]
fn origin_translation_real_sync_retains_sleeping_contacts_through_repeated_shifts() {
    let mut app = app(false);
    #[cfg(feature = "dim2")]
    let ground = Collider::cuboid(10., 0.5);
    #[cfg(feature = "dim3")]
    let ground = Collider::cuboid(10., 0.5, 10.);
    app.world_mut()
        .spawn((Actor, ground, Transform::from_xyz(0., -0.5, 0.)));
    let actor = app
        .world_mut()
        .spawn((
            Actor,
            RigidBody::Dynamic,
            Collider::ball(0.5),
            Transform::from_xyz(0., 0.5, 0.),
            TransformInterpolation::default(),
        ))
        .id();
    for _ in 0..360 {
        app.update();
    }
    assert!(body_state(&mut app, actor).3);
    let original = body_state(&mut app, actor).0;
    let mut sum = Vect::ZERO;
    for delta in [v(32., -16., 8.), v(-64., 32., -16.), v(32., -16., 8.)] {
        app.world_mut().resource_mut::<Request>().0 = Some(delta);
        app.update();
        assert_eq!(app.world().resource::<Outcome>().0, Some(Ok(())));
        sum += delta;
        for _ in 0..20 {
            app.update();
        }
        let (pose, _, _, sleeping) = body_state(&mut app, actor);
        assert!(
            sleeping,
            "SyncBackend treated an origin shift as physical motion"
        );
        assert!((pose - (original + sum)).length() < 0.0001);
        let world = app.world_mut();
        let set = world.query::<&RapierRigidBodySet>().single(world).unwrap();
        assert_eq!(
            set.last_body_transform_set[&set.entity2body[&actor]],
            *world.get::<GlobalTransform>(actor).unwrap()
        );
        let simulation = world
            .query::<&RapierContextSimulation>()
            .single(world)
            .unwrap();
        assert!(simulation
            .narrow_phase
            .contact_pairs()
            .any(|pair| pair.has_any_active_contact()));
    }
}

#[test]
fn origin_translation_interpolated_pending_kinematic_motion_matches_unshifted_app() {
    let build = || {
        let mut app = app(true);
        let actor = app
            .world_mut()
            .spawn((
                Actor,
                RigidBody::KinematicPositionBased,
                Collider::ball(0.25),
                Transform::from_xyz(2., 3., 0.),
                TransformInterpolation::default(),
            ))
            .id();
        for _ in 0..4 {
            app.update();
        }
        (app, actor)
    };
    let (mut moved, a) = build();
    let (mut control, b) = build();
    moved
        .world_mut()
        .get_mut::<Transform>(a)
        .unwrap()
        .translation
        .x += 0.5;
    control
        .world_mut()
        .get_mut::<Transform>(b)
        .unwrap()
        .translation
        .x += 0.5;
    let delta = v(32., -16., 8.);
    moved.world_mut().resource_mut::<Request>().0 = Some(delta);
    for frame in 0..6 {
        moved.update();
        control.update();
        assert_eq!(moved.world().resource::<Outcome>().0, Some(Ok(())));
        let (pos, next, velocity, _) = body_state(&mut moved, a);
        let (ref_pos, ref_next, ref_velocity, _) = body_state(&mut control, b);
        assert!((pos - (ref_pos + delta)).length() < 0.0001, "frame {frame}");
        assert!((next - (ref_next + delta)).length() < 0.0001);
        assert!((velocity - ref_velocity).length() < 0.0001);
        assert!(
            (moved.world().get::<Transform>(a).unwrap().translation
                - (control.world().get::<Transform>(b).unwrap().translation + render_delta(delta)))
            .length()
                < 0.0001
        );
    }
}

#[test]
fn origin_translation_bad_interpolation_refuses_whole_fixture_before_context_mutation() {
    let mut app = app(false);
    let actor = app
        .world_mut()
        .spawn((
            Actor,
            RigidBody::Dynamic,
            Collider::ball(0.5),
            Transform::default(),
            TransformInterpolation::default(),
        ))
        .id();
    app.update();
    let bad = TransformInterpolation {
        start: Some(Pose::IDENTITY),
        end: Some(Pose::from_translation(v(f32::MAX, 0., 0.))),
    };
    app.world_mut().entity_mut(actor).insert(bad);
    let before = {
        let world = app.world_mut();
        snapshot(world.query::<&RapierRigidBodySet>().single(world).unwrap())
    };
    let transform = *app.world().get::<Transform>(actor).unwrap();
    app.world_mut().resource_mut::<Request>().0 = Some(v(f32::MAX, 0., 0.));
    app.world_mut().run_system_once(apply_request).unwrap();
    assert_eq!(
        app.world().resource::<Outcome>().0,
        Some(Err(OriginTranslationError::NonFiniteState))
    );
    let world = app.world_mut();
    assert_eq!(
        snapshot(world.query::<&RapierRigidBodySet>().single(world).unwrap()),
        before
    );
    assert_eq!(*world.get::<Transform>(actor).unwrap(), transform);
    assert_eq!(*world.get::<TransformInterpolation>(actor).unwrap(), bad);
}
