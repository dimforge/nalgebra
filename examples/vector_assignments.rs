use nalgebra::{DVector, SVector};

fn main() {
    // Set full SVector (SV) to DVector (DV)
    let (sv, mut dv) = (new_sv(), new_dv());
    dv.copy_from(&sv);
    assert_eq!(dv, SVector::<f32, 5>::new(0.0, 1.0, 2.0, 3.0, 4.0));

    // Set full DV to SV
    let (mut sv, dv) = (new_sv(), new_dv());
    sv.copy_from(&dv);
    assert_eq!(sv, SVector::<f32, 5>::new(0.0, 0.1, 0.2, 0.3, 0.4));

    // Create new FV from another FV
    let sv = new_sv();
    let sv2: SVector<f32, 5> = sv; // SVector is Copy
    assert_eq!(sv2, sv);

    // Create new FV from another DV
    let dv = new_dv();
    let sv2: SVector<f32, 5> = dv.fixed_view::<5, 1>(0, 0).into();
    assert_eq!(sv2, dv);

    // Create new DV from another FV
    let sv = new_sv();
    let dv2: DVector<f32> = DVector::from_row_slice(sv.as_slice());
    assert_eq!(dv2, sv);

    // Create new DV from another DV
    let dv = new_dv();
    let dv2: DVector<f32> = dv.clone(); // DVector is only Clone
    assert_eq!(dv2, dv);

    // Set part of FV to another FV
    let (mut sv, sv_sub) = (new_sv(), new_sv_sub());
    *sv.fixed_view_mut::<3, 1>(1, 0) = *sv_sub;
    assert_eq!(sv, SVector::<f32, 5>::new(0.0, 0.5, 1.5, 2.5, 4.0));

    // Set part of FV to part of another FV
    let (mut sv, sv_sub) = (new_sv(), new_sv_sub());
    *sv.fixed_view_mut::<2, 1>(2, 0) = *sv_sub.fixed_view::<2, 1>(1, 0);
    assert_eq!(sv, SVector::<f32, 5>::new(0.0, 1.0, 1.5, 2.5, 4.0));

    // Set part of FV to another DV
    let (mut sv, dv_sub) = (new_sv(), new_dv_sub());
    *sv.fixed_view_mut::<3, 1>(1, 0) = *dv_sub.fixed_view::<3, 1>(0, 0);
    assert_eq!(sv, SVector::<f32, 5>::new(0.0, 0.5, 1.5, 2.5, 4.0));

    // Set part of FV to part of another DV
    let (mut sv, dv_sub) = (new_sv(), new_dv_sub());
    *sv.fixed_view_mut::<2, 1>(2, 0) = *dv_sub.fixed_view::<2, 1>(1, 0);
    assert_eq!(sv, SVector::<f32, 5>::new(0.0, 1.0, 1.5, 2.5, 4.0));

    // Set part of DV to another FV
    let (mut dv, sv_sub) = (new_dv(), new_sv_sub());
    dv.view_mut((1, 0), (3, 1)).copy_from(&sv_sub);
    assert_eq!(dv, SVector::<f32, 5>::new(0.0, 0.5, 1.5, 2.5, 0.4));

    // Set part of DV to part of another FV
    let (mut dv, sv_sub) = (new_dv(), new_sv_sub());
    dv.view_mut((2, 0), (2, 1))
        .copy_from(&sv_sub.fixed_view::<2, 1>(1, 0));
    assert_eq!(dv, SVector::<f32, 5>::new(0.0, 0.1, 1.5, 2.5, 0.4));

    // Set part of DV to another DV
    let (mut dv, dv_sub) = (new_dv(), new_dv_sub());
    dv.view_mut((1, 0), (3, 1)).copy_from(&dv_sub);
    assert_eq!(dv, SVector::<f32, 5>::new(0.0, 0.5, 1.5, 2.5, 0.4));

    // Set part of DV to part of another DV
    let (mut dv, dv_sub) = (new_dv(), new_dv_sub());
    dv.view_mut((2, 0), (2, 1))
        .copy_from(&dv_sub.view((1, 0), (2, 1)));
    assert_eq!(dv, SVector::<f32, 5>::new(0.0, 0.1, 1.5, 2.5, 0.4));

    // Set part of DV to part of another DV (alternative)
    let (mut dv, dv_sub) = (new_dv(), new_dv_sub());
    dv.view_mut((2, 0), (2, 1))
        .copy_from(&dv_sub.fixed_view::<2, 1>(1, 0));
    assert_eq!(dv, SVector::<f32, 5>::new(0.0, 0.1, 1.5, 2.5, 0.4));
}

// Create an example fixed sized vector with length 5
fn new_sv() -> SVector<f32, 5> {
    SVector::<f32, 5>::new(0.0, 1.0, 2.0, 3.0, 4.0)
}

// Create an example dynamically sized vector with length 5
fn new_dv() -> DVector<f32> {
    DVector::from_row_slice(&[0.0, 0.1, 0.2, 0.3, 0.4])
}

// Create an example fixed sized vector with length 3
fn new_sv_sub() -> SVector<f32, 3> {
    SVector::<f32, 3>::new(0.5, 1.5, 2.5)
}

// Create an example dynamically sized vector with length 3
fn new_dv_sub() -> DVector<f32> {
    DVector::from_row_slice(&[0.5, 1.5, 2.5])
}
