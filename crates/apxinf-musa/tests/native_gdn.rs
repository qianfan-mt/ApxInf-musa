#![cfg(feature = "mttx")]

use apxinf_musa::native_gdn::NativeGdnLauncher;

#[test]
fn native_gdn_is_opt_in_and_missing_library_is_reported() {
    std::env::remove_var("APXINF_MUSA_NATIVE_GDN_LIB");
    assert!(NativeGdnLauncher::from_env(0).unwrap().is_none());

    std::env::set_var("APXINF_MUSA_NATIVE_GDN_LIB", "/definitely/missing/libmate_native.so");
    let error = match NativeGdnLauncher::from_env(0) {
        Err(error) => error,
        Ok(_) => panic!("missing native GDN library unexpectedly loaded"),
    };
    assert!(error.contains("libmate_native.so"), "unexpected loader error: {error}");
    std::env::remove_var("APXINF_MUSA_NATIVE_GDN_LIB");
}

#[test]
fn native_gdn_loader_reports_missing_symbol() {
    let path = std::env::var("APXINF_MUSA_NATIVE_GDN_TEST_LIB");
    let Ok(path) = path else { return };
    std::env::set_var("APXINF_MUSA_NATIVE_GDN_LIB", path);
    let error = match NativeGdnLauncher::from_env(0) {
        Err(error) => error,
        Ok(_) => panic!("library without native GDN symbol unexpectedly loaded"),
    };
    assert!(error.contains("mate_native_launch"), "unexpected symbol error: {error}");
    std::env::remove_var("APXINF_MUSA_NATIVE_GDN_LIB");
}

#[test]
fn native_gdn_loader_can_be_disabled_without_a_library() {
    std::env::remove_var("APXINF_MUSA_NATIVE_GDN_LIB");
    assert!(NativeGdnLauncher::from_env(2).unwrap().is_none());
}

#[test]
fn native_gdn_loader_does_not_claim_a_missing_file_is_available() {
    std::env::set_var("APXINF_MUSA_NATIVE_GDN_LIB", "/tmp/not-a-real-native-gdn.so");
    assert!(NativeGdnLauncher::from_env(0).is_err());
    std::env::remove_var("APXINF_MUSA_NATIVE_GDN_LIB");
}
