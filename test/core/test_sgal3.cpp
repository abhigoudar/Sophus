#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "sophus/sgal3.hpp"  // Path to your SGal3 header

namespace {

// Test fixture for SGal3 tests
class SGal3Test : public ::testing::Test {
protected:
    using Scalar = double;
    using SGal3d = Sophus::SGal3<Scalar>;
    using TangentVector = SGal3d::TangentVector;
    
    // Tolerance for floating point comparisons
    static constexpr Scalar kTolerance = 1e-10;
    
    void SetUp() override {
        // Create test elements
        // Identity rotation
        Sophus::SO3<Scalar> C_identity;
        
        // Non-identity rotation (90 degrees around z-axis)
        Eigen::AngleAxisd angle_axis(M_PI / 2.0, Eigen::Vector3d(0, 0, 1));
        Sophus::SO3<Scalar> C_rotated(angle_axis.toRotationMatrix());
        
        // Translation vector
        Eigen::Matrix<Scalar, 3, 1> translation(1.0, 2.0, 3.0);
        
        // Boost (velocity) vector
        Eigen::Matrix<Scalar, 3, 1> boost(0.5, 0.2, 0.1);
        
        // Timestamp
        Eigen::Matrix<Scalar, 1, 1> timestamp;
        timestamp << 1.5;
        
        // Create identity and non-identity elements
        G_identity = SGal3d();
        G_element = SGal3d(C_rotated, translation, boost, timestamp);
    }
    
    SGal3d G_identity;
    SGal3d G_element;
};

// ============================================================================
// Test 1: G * G.inverse() is an identity element
// ============================================================================

TEST_F(SGal3Test, GroupCompositionWithInverseGivesIdentity) {
    // Test with identity element
    SGal3d result_identity = G_identity * G_identity.inverse();
    
    // The result should be identity
    EXPECT_NEAR(result_identity.rotation().matrix()(0, 0), 1.0, kTolerance);
    EXPECT_NEAR(result_identity.rotation().matrix()(1, 1), 1.0, kTolerance);
    EXPECT_NEAR(result_identity.rotation().matrix()(2, 2), 1.0, kTolerance);
    EXPECT_TRUE(result_identity.translation().isZero(kTolerance));
    EXPECT_TRUE(result_identity.boost().isZero(kTolerance));
    EXPECT_NEAR(result_identity.timestamp()(0, 0), 0.0, kTolerance);
    
    // Test with non-identity element
    SGal3d result = G_element * G_element.inverse();
    
    // Check rotation part: should be identity
    Eigen::Matrix3d rot_result = result.rotation().matrix();
    Eigen::Matrix3d rot_identity = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(rot_result(i, j), rot_identity(i, j), kTolerance)
                << "Rotation matrix element (" << i << ", " << j << ") mismatch";
        }
    }
    
    // Check translation part: should be zero
    EXPECT_TRUE(result.translation().isZero(kTolerance))
        << "Translation should be zero, got: " << result.translation().transpose();
    
    // Check boost part: should be zero
    EXPECT_TRUE(result.boost().isZero(kTolerance))
        << "Boost should be zero, got: " << result.boost().transpose();
    
    // Check timestamp part: should be zero
    EXPECT_NEAR(result.timestamp()(0, 0), 0.0, kTolerance)
        << "Timestamp should be zero, got: " << result.timestamp()(0, 0);
}

// Test multiple elements to ensure the property holds generally
TEST_F(SGal3Test, GroupCompositionWithInverseGivesIdentityMultipleElements) {
    // Create several random elements
    std::vector<SGal3d> elements;
    
    for (int i = 0; i < 5; ++i) {
        // Random rotation
        Eigen::Vector3d omega = Eigen::Vector3d::Random() * 0.5;
        Sophus::SO3<Scalar> C = Sophus::SO3<Scalar>::exp(omega);
        
        // Random translation
        Eigen::Matrix<Scalar, 3, 1> p = Eigen::Matrix<Scalar, 3, 1>::Random();
        
        // Random boost
        Eigen::Matrix<Scalar, 3, 1> v = Eigen::Matrix<Scalar, 3, 1>::Random() * 0.5;
        
        // Random timestamp
        Eigen::Matrix<Scalar, 1, 1> t;
        t << (Scalar(i) + 1.0);
        
        elements.push_back(SGal3d(C, p, v, t));
    }
    
    // Test each element
    for (const auto& G : elements) {
        SGal3d result = G * G.inverse();
        
        // Check that result is identity
        Eigen::Matrix3d rot = result.rotation().matrix();
        EXPECT_TRUE(rot.isApprox(Eigen::Matrix3d::Identity(), kTolerance));
        EXPECT_TRUE(result.translation().isZero(kTolerance));
        EXPECT_TRUE(result.boost().isZero(kTolerance));
        EXPECT_NEAR(result.timestamp()(0, 0), 0.0, kTolerance);
    }
}

// ============================================================================
// Test 2: g - log(exp(g)) is close to zero
// ============================================================================

TEST_F(SGal3Test, TangentVectorExponentialLogarithmRoundTrip) {
    // Create a tangent vector
    TangentVector g;
    g << 0.1, 0.2, 0.3,      // rho (translation)
         0.05, 0.1, 0.15,     // nu (velocity)
         0.01, 0.02, 0.03,    // theta (rotation)
         0.5;                  // iota (time)
    
    // Compute exp(g)
    SGal3d G = SGal3d::exp(g);
    
    // Compute log(exp(g))
    TangentVector g_recovered = G.log();
    
    // Compute the difference
    TangentVector diff = g - g_recovered;
    
    // Check that difference is close to zero
    for (int i = 0; i < TangentVector::RowsAtCompileTime; ++i) {
        EXPECT_NEAR(diff(i), 0.0, kTolerance)
            << "Component " << i << " of tangent vector differs: "
            << "g[" << i << "] = " << g(i) << ", "
            << "g_recovered[" << i << "] = " << g_recovered(i);
    }
    
    // Check the norm of the difference
    EXPECT_NEAR(diff.norm(), 0.0, kTolerance)
        << "Tangent vector difference norm: " << diff.norm();
}

// Test with multiple tangent vectors with different magnitudes
TEST_F(SGal3Test, TangentVectorExponentialLogarithmRoundTripMultiple) {
    // Test vectors with different magnitudes
    std::vector<Scalar> magnitudes = {0.01, 0.1, 0.5, 1.0};
    
    for (Scalar mag : magnitudes) {
        TangentVector g = TangentVector::Random() * mag;
        
        // Ensure time component is non-zero
        g(9) = mag * 0.5 + 0.1;
        
        // Compute exp(g)
        SGal3d G = SGal3d::exp(g);
        
        // Compute log(exp(g))
        TangentVector g_recovered = G.log();
        
        // Compute the difference
        TangentVector diff = g - g_recovered;
        
        // Check that difference is close to zero with relaxed tolerance for larger magnitudes
        Scalar tolerance = kTolerance * (1.0 + mag);
        EXPECT_LT(diff.norm(), tolerance)
            << "Magnitude: " << mag << ", "
            << "Difference norm: " << diff.norm() << ", "
            << "Tolerance: " << tolerance;
    }
}

// Test with small perturbations (important for optimization algorithms)
TEST_F(SGal3Test, TangentVectorExponentialLogarithmSmallPerturbations) {
    // Small tangent vector (important for iterative optimization)
    TangentVector g;
    g << 1e-6, 2e-6, 3e-6,
         5e-7, 1e-6, 1.5e-6,
         1e-7, 2e-7, 3e-7,
         5e-7;
    
    SGal3d G = SGal3d::exp(g);
    TangentVector g_recovered = G.log();
    TangentVector diff = g - g_recovered;
    
    // For small perturbations, we need higher relative accuracy
    EXPECT_NEAR(diff.norm(), 0.0, kTolerance * 10.0)
        << "Small perturbation test failed. Difference norm: " << diff.norm();
}

// ============================================================================
// Test 3: Map<SGal3d> preserves data when converting from SGal3d and back
// ============================================================================

TEST_F(SGal3Test, MapPreservesDataRoundTrip) {
    // Create an original element
    Sophus::SO3<Scalar> C_original;
    {
        Eigen::AngleAxisd angle_axis(M_PI / 3.0, Eigen::Vector3d(1, 1, 1).normalized());
        C_original = Sophus::SO3<Scalar>(angle_axis.toRotationMatrix());
    }
    
    Eigen::Matrix<Scalar, 3, 1> p_original(1.5, 2.5, 3.5);
    Eigen::Matrix<Scalar, 3, 1> v_original(0.3, 0.4, 0.5);
    Eigen::Matrix<Scalar, 1, 1> t_original;
    t_original << 2.5;
    
    SGal3d G_original(C_original, p_original, v_original, t_original);
    
    // Create a buffer and map it
    std::vector<Scalar> buffer(SGal3d::num_parameters);
    
    // Copy G_original data to buffer
    // Sophus::SO3<Scalar>::num_parameters_is_4_if_quaternion;
    Eigen::Map<Sophus::SO3<Scalar>>(buffer.data()).setQuaternion(
        G_original.rotation().unit_quaternion());
    std::copy(G_original.translation().data(), 
              G_original.translation().data() + 3,
              buffer.data() + Sophus::SO3<Scalar>::num_parameters);
    std::copy(G_original.boost().data(),
              G_original.boost().data() + 3,
              buffer.data() + Sophus::SO3<Scalar>::num_parameters + 3);
    std::copy(G_original.timestamp().data(),
              G_original.timestamp().data() + 1,
              buffer.data() + Sophus::SO3<Scalar>::num_parameters + 6);
    
    // Create a Map from the buffer
    Eigen::Map<Sophus::SGal3<Scalar>> G_mapped(buffer.data());
    
    // Create a new SGal3d from the mapped data
    SGal3d G_recovered(
        G_mapped.rotation(),
        G_mapped.translation(),
        G_mapped.boost(),
        G_mapped.timestamp()
    );
    
    // Compare original and recovered
    // Rotation comparison
    Eigen::Matrix3d rot_diff = G_original.rotation().matrix() - G_recovered.rotation().matrix();
    EXPECT_LT(rot_diff.norm(), kTolerance)
        << "Rotation matrices differ. Difference norm: " << rot_diff.norm();
    
    // Translation comparison
    Eigen::Matrix<Scalar, 3, 1> p_diff = G_original.translation() - G_recovered.translation();
    EXPECT_LT(p_diff.norm(), kTolerance)
        << "Translation vectors differ: "
        << "original = " << G_original.translation().transpose() << ", "
        << "recovered = " << G_recovered.translation().transpose();
    
    // Boost comparison
    Eigen::Matrix<Scalar, 3, 1> v_diff = G_original.boost() - G_recovered.boost();
    EXPECT_LT(v_diff.norm(), kTolerance)
        << "Boost vectors differ: "
        << "original = " << G_original.boost().transpose() << ", "
        << "recovered = " << G_recovered.boost().transpose();
    
    // Timestamp comparison
    EXPECT_NEAR(G_original.timestamp()(0), G_recovered.timestamp()(0), kTolerance)
        << "Timestamps differ: "
        << "original = " << G_original.timestamp()(0) << ", "
        << "recovered = " << G_recovered.timestamp()(0);
}

// Test Map with multiple elements
TEST_F(SGal3Test, MapPreservesDataMultipleElements) {
    std::vector<SGal3d> originals;
    
    // Create several test elements
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d omega = Eigen::Vector3d::Random() * 0.3;
        Sophus::SO3<Scalar> C = Sophus::SO3<Scalar>::exp(omega);
        Eigen::Matrix<Scalar, 3, 1> p = Eigen::Matrix<Scalar, 3, 1>::Random() * 2.0;
        Eigen::Matrix<Scalar, 3, 1> v = Eigen::Matrix<Scalar, 3, 1>::Random() * 0.5;
        Eigen::Matrix<Scalar, 1, 1> t;
        t << (Scalar(i) + 1.0);
        
        originals.push_back(SGal3d(C, p, v, t));
    }
    
    for (const auto& G_original : originals) {
        // Create buffer and fill it
        std::vector<Scalar> buffer(SGal3d::num_parameters);
        
        // Note: This is a simplified version. In production, use proper serialization
        Eigen::Map<Sophus::SO3<Scalar>>(buffer.data()).setQuaternion(
            G_original.rotation().unit_quaternion());
        std::copy(G_original.translation().data(),
                  G_original.translation().data() + 3,
                  buffer.data() + Sophus::SO3<Scalar>::num_parameters);
        std::copy(G_original.boost().data(),
                  G_original.boost().data() + 3,
                  buffer.data() + Sophus::SO3<Scalar>::num_parameters + 3);
        std::copy(G_original.timestamp().data(),
                  G_original.timestamp().data() + 1,
                  buffer.data() + Sophus::SO3<Scalar>::num_parameters + 6);
        
        // Create Map and verify
        Eigen::Map<Sophus::SGal3<Scalar>> G_mapped(buffer.data());
        SGal3d G_recovered(
            G_mapped.rotation(),
            G_mapped.translation(),
            G_mapped.boost(),
            G_mapped.timestamp()
        );
        
        // Verify all components match
        EXPECT_TRUE(G_original.rotation().matrix().isApprox(
            G_recovered.rotation().matrix(), kTolerance));
        EXPECT_TRUE(G_original.translation().isApprox(
            G_recovered.translation(), kTolerance));
        EXPECT_TRUE(G_original.boost().isApprox(
            G_recovered.boost(), kTolerance));
        EXPECT_NEAR(G_original.timestamp()(0), G_recovered.timestamp()(0), kTolerance);
    }
}

// Test Map modification reflects in original buffer
TEST_F(SGal3Test, MapModificationAffectsBuffer) {
    // Create a buffer
    std::vector<Scalar> buffer(Sophus::SGal3<Scalar>::num_parameters);
    
    // Initialize with an identity element
    Sophus::SGal3<Scalar> G_identity;
    Eigen::Map<Sophus::SO3<Scalar>>(buffer.data()).setQuaternion(
        G_identity.rotation().unit_quaternion());
    std::fill(buffer.begin() + Sophus::SO3<Scalar>::num_parameters, buffer.end(), 0.0);
    
    // Create a Map
    Eigen::Map<Sophus::SGal3<Scalar>> G_map(buffer.data());
    
    // Modify through the Map
    Eigen::Vector3d new_translation(1.0, 2.0, 3.0);
    G_map.translation() = new_translation;
    
    Eigen::Vector3d new_boost(0.1, 0.2, 0.3);
    G_map.boost() = new_boost;
    
    // Check that the buffer was modified
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(buffer[Sophus::SO3<Scalar>::num_parameters + i],
                    new_translation(i), kTolerance);
        EXPECT_NEAR(buffer[Sophus::SO3<Scalar>::num_parameters + 3 + i],
                    new_boost(i), kTolerance);
    }
}

// ============================================================================
// Additional Consistency Tests
// ============================================================================

TEST_F(SGal3Test, IdentityElementProperties) {
    SGal3d identity;
    
    // Identity rotation should be identity matrix
    EXPECT_TRUE(identity.rotation().matrix().isApprox(
        Eigen::Matrix3d::Identity(), kTolerance));
    
    // Translation, boost, and timestamp should be zero
    EXPECT_TRUE(identity.translation().isZero(kTolerance));
    EXPECT_TRUE(identity.boost().isZero(kTolerance));
    EXPECT_NEAR(identity.timestamp()(0), 0.0, kTolerance);
    
    // G * identity = G
    SGal3d result = G_element * identity;
    EXPECT_TRUE(result.rotation().matrix().isApprox(
        G_element.rotation().matrix(), kTolerance));
    EXPECT_TRUE(result.translation().isApprox(
        G_element.translation(), kTolerance));
    EXPECT_TRUE(result.boost().isApprox(
        G_element.boost(), kTolerance));
    EXPECT_NEAR(result.timestamp()(0), G_element.timestamp()(0), kTolerance);
}

TEST_F(SGal3Test, AssignmentOperator) {
    SGal3d G1 = G_element;
    SGal3d G2;
    
    // Test assignment from G1 to G2
    G2 = G1;
    
    EXPECT_TRUE(G2.rotation().matrix().isApprox(
        G1.rotation().matrix(), kTolerance));
    EXPECT_TRUE(G2.translation().isApprox(G1.translation(), kTolerance));
    EXPECT_TRUE(G2.boost().isApprox(G1.boost(), kTolerance));
    EXPECT_NEAR(G2.timestamp()(0), G1.timestamp()(0), kTolerance);
}

}  // namespace

// Main function for running tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}