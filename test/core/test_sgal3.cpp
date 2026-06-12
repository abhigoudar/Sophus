#include <algorithm>
#include <iostream>
#include <vector>

#include <sophus/sgal3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.

namespace Eigen {
template class Map<Sophus::SGal3<double>>;
template class Map<Sophus::SGal3<double> const>;
}  // namespace Eigen

namespace Sophus {

template class SGal3<double>;
template class SGal3<float>;
#if SOPHUS_CERES
template class SGal3<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
class Tests {
 public:
  using SGal3Type = SGal3<Scalar>;
  using SO3Type = SO3<Scalar>;
  using Point = Vector3<Scalar>;
  using Tangent = typename SGal3<Scalar>::TangentVector;
  // SGal3 carries a scalar time-stamp, represented as a 1x1 vector.
  using Time = Eigen::Matrix<Scalar, 1, 1>;
  Scalar const kPi = Constants<Scalar>::pi();

  Tests() {
    sgal3_vec_ = getTestSGal3s();

    // Tangent layout: [rho (translation, 3), nu (boost, 3),
    //                  theta (rotation, 3), iota (time, 1)].
    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
        Scalar(0), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
        Scalar(0), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(0), Scalar(1), Scalar(0), Scalar(0),
        Scalar(0), Scalar(0), Scalar(0), Scalar(0.5);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(-5), Scalar(10), Scalar(0), Scalar(0), Scalar(0),
        Scalar(0.1), Scalar(0.2), Scalar(0.3), Scalar(1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(0), Scalar(0.5), Scalar(0.2),
        Scalar(0.1), Scalar(0), Scalar(0), Scalar(1), Scalar(0.2);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(2), Scalar(-1), Scalar(0), Scalar(-1), Scalar(1), Scalar(0),
        Scalar(0.1), Scalar(0.05), Scalar(0.15), Scalar(0.7);
    tangent_vec_.push_back(tmp);
  }

  void runAll() {
    bool passed = testGroupOperations();
    passed &= testExpLog();
    passed &= testExpLogRandom();
    passed &= testRawDataAcces();
    passed &= testMutatingAccessors();
    passed &= testConstructors();
    processTestResult(passed);
  }

 private:
  // Builds a set of representative SGal3 elements to exercise the tests.
  std::vector<SGal3Type, Eigen::aligned_allocator<SGal3Type>> getTestSGal3s() {
    std::vector<SGal3Type, Eigen::aligned_allocator<SGal3Type>> sgal3_vec;

    SO3Type const C0;  // identity rotation
    SO3Type const C1 =
        SO3Type::exp(Vector3<Scalar>(Scalar(0.2), Scalar(0.5), Scalar(0.1)));
    SO3Type const C2 =
        SO3Type::exp(Vector3<Scalar>(Scalar(0), Scalar(0), Scalar(kPi / 2)));

    Point const p0 = Point::Zero();
    Point const p1(Scalar(1), Scalar(2), Scalar(3));
    Point const v0 = Point::Zero();
    Point const v1(Scalar(0.3), Scalar(-0.2), Scalar(0.5));
    Time const t0 = Time::Zero();
    Time t1;
    t1 << Scalar(1.5);

    sgal3_vec.push_back(SGal3Type());                  // identity
    sgal3_vec.push_back(SGal3Type(C1, p0, v0, t0));    // rotation only
    sgal3_vec.push_back(SGal3Type(C0, p1, v0, t0));    // translation only
    sgal3_vec.push_back(SGal3Type(C0, p0, v1, t0));    // boost only
    sgal3_vec.push_back(SGal3Type(C0, p0, v0, t1));    // time only
    sgal3_vec.push_back(SGal3Type(C2, p1, v1, t1));    // general element
    sgal3_vec.push_back(SGal3Type(C1, -p1, -v1, t1));  // general element

    return sgal3_vec;
  }

  // Copies a group element into a contiguous raw parameter buffer using the
  // ``[quaternion, translation, boost, timestamp]`` storage order.
  template <class Derived>
  static void toRawBuffer(SGal3Base<Derived> const& G, Scalar* buffer) {
    Eigen::Map<SO3Type>(buffer).setQuaternion(G.rotation().unit_quaternion());
    std::copy(G.translation().data(), G.translation().data() + 3,
              buffer + SO3Type::num_parameters);
    std::copy(G.boost().data(), G.boost().data() + 3,
              buffer + SO3Type::num_parameters + 3);
    std::copy(G.timestamp().data(), G.timestamp().data() + 1,
              buffer + SO3Type::num_parameters + 6);
  }

  // Component-wise approximate comparison of two SGal3 elements. Templated on
  // the CRTP-derived types so that it works for SGal3<Scalar> as well as for
  // Eigen::Map<SGal3<Scalar>> and Eigen::Map<SGal3<Scalar> const>.
  template <class LhsDerived, class RhsDerived>
  void expectSame(bool& passed, SGal3Base<LhsDerived> const& lhs,
                  SGal3Base<RhsDerived> const& rhs, Scalar prec,
                  char const* msg) {
    SOPHUS_TEST_APPROX(passed, lhs.rotation().matrix(), rhs.rotation().matrix(),
                       prec, "{} (rotation)", msg);
    SOPHUS_TEST_APPROX(passed, lhs.translation().eval(),
                       rhs.translation().eval(), prec, "{} (translation)", msg);
    SOPHUS_TEST_APPROX(passed, lhs.boost().eval(), rhs.boost().eval(), prec,
                       "{} (boost)", msg);
    SOPHUS_TEST_APPROX(passed, lhs.timestamp()(0, 0), rhs.timestamp()(0, 0),
                       prec, "{} (timestamp)", msg);
  }

  // G * G.inverse() == identity, and G * identity == identity * G == G.
  bool testGroupOperations() {
    bool passed = true;
    Scalar const eps = Constants<Scalar>::epsilon();
    SGal3Type const identity;

    for (size_t i = 0; i < sgal3_vec_.size(); ++i) {
      SGal3Type const& G = sgal3_vec_[i];

      SGal3Type const should_be_identity = G * G.inverse();
      SOPHUS_TEST_APPROX(passed, should_be_identity.rotation().matrix(),
                         Matrix3<Scalar>::Identity().eval(), eps,
                         "G * G.inverse() rotation case: {}", i);
      SOPHUS_TEST_APPROX(passed, should_be_identity.translation().eval(),
                         Point::Zero().eval(), eps,
                         "G * G.inverse() translation case: {}", i);
      SOPHUS_TEST_APPROX(passed, should_be_identity.boost().eval(),
                         Point::Zero().eval(), eps,
                         "G * G.inverse() boost case: {}", i);
      SOPHUS_TEST_APPROX(passed, should_be_identity.timestamp()(0, 0),
                         Scalar(0), eps,
                         "G * G.inverse() timestamp case: {}", i);

      expectSame(passed, G * identity, G, eps, "G * identity == G");
      expectSame(passed, identity * G, G, eps, "identity * G == G");
    }
    return passed;
  }

  // g - log(exp(g)) is close to zero (and the group-side round trip).
  bool testExpLog() {
    bool passed = true;
    Scalar const prec = Constants<Scalar>::epsilonSqrt();

    for (size_t i = 0; i < tangent_vec_.size(); ++i) {
      Tangent const& g = tangent_vec_[i];
      Tangent const g_recovered = SGal3Type::exp(g).log();
      SOPHUS_TEST_APPROX(passed, g_recovered, g, prec,
                         "g - log(exp(g)) case: {}", i);
    }

    // Round trip starting from the group: G - exp(log(G)).
    for (size_t i = 0; i < sgal3_vec_.size(); ++i) {
      SGal3Type const& G = sgal3_vec_[i];
      SGal3Type const G_recovered = SGal3Type::exp(G.log());
      expectSame(passed, G_recovered, G, prec, "G - exp(log(G))");
    }

    // Small perturbations (important for iterative optimization).
    Tangent small;
    small << Scalar(1e-6), Scalar(2e-6), Scalar(3e-6), Scalar(5e-7),
        Scalar(1e-6), Scalar(1.5e-6), Scalar(1e-7), Scalar(2e-7), Scalar(3e-7),
        Scalar(5e-7);
    Tangent const small_recovered = SGal3Type::exp(small).log();
    SOPHUS_TEST_APPROX(passed, small_recovered, small, prec,
                       "g - log(exp(g)) small perturbation");

    return passed;
  }

  // exp/log round trip for random tangent vectors of varying magnitude.
  // Guarded to floating-point scalars (random sampling is not meaningful for
  // e.g. ceres::Jet), mirroring the SFINAE pattern used in test_se3.cpp.
  template <class S = Scalar>
  std::enable_if_t<std::is_floating_point<S>::value, bool> testExpLogRandom() {
    bool passed = true;
    Scalar const base = Scalar(10) * Constants<Scalar>::epsilonSqrt();

    for (Scalar const mag :
         {Scalar(0.01), Scalar(0.1), Scalar(0.5), Scalar(1.0)}) {
      for (int i = 0; i < 25; ++i) {
        Tangent g = Tangent::Random() * mag;
        g(9) = mag * Scalar(0.5) + Scalar(0.1);  // non-zero time component
        Tangent const g_recovered = SGal3Type::exp(g).log();
        SOPHUS_TEST_APPROX(passed, g_recovered, g, base * (Scalar(1) + mag),
                           "random g - log(exp(g)), magnitude {}", mag);
      }
    }
    return passed;
  }

  template <class S = Scalar>
  std::enable_if_t<!std::is_floating_point<S>::value, bool> testExpLogRandom() {
    return true;
  }

  bool testRawDataAcces() {
    bool passed = true;
    Scalar const eps = Constants<Scalar>::epsilon();
    int const kQuat = SO3Type::num_parameters;  // == 4

    // Raw parameter layout: [quaternion (4), translation (3), boost (3),
    //                        timestamp (1)] == num_parameters (== 11).
    Eigen::Matrix<Scalar, SGal3Type::num_parameters, 1> raw;
    raw << Scalar(0), Scalar(0), Scalar(0), Scalar(1),  // identity quaternion
        Scalar(1), Scalar(2), Scalar(3),                // translation
        Scalar(0.1), Scalar(0.2), Scalar(0.3),          // boost
        Scalar(1.5);                                    // timestamp

    Eigen::Map<SGal3Type const> map_of_const_sgal3(raw.data());
    SOPHUS_TEST_APPROX(
        passed, map_of_const_sgal3.rotation().unit_quaternion().coeffs().eval(),
        raw.template head<4>().eval(), eps, "");
    SOPHUS_TEST_APPROX(passed, map_of_const_sgal3.translation().eval(),
                       raw.template segment<3>(kQuat).eval(), eps, "");
    SOPHUS_TEST_APPROX(passed, map_of_const_sgal3.boost().eval(),
                       raw.template segment<3>(kQuat + 3).eval(), eps, "");
    SOPHUS_TEST_APPROX(passed, map_of_const_sgal3.timestamp()(0, 0),
                       raw(kQuat + 6), eps, "");

    SOPHUS_TEST_EQUAL(
        passed, map_of_const_sgal3.rotation().unit_quaternion().coeffs().data(),
        raw.data(), "");
    SOPHUS_TEST_EQUAL(passed, map_of_const_sgal3.translation().data(),
                      raw.data() + kQuat, "");
    SOPHUS_TEST_EQUAL(passed, map_of_const_sgal3.boost().data(),
                      raw.data() + kQuat + 3, "");
    SOPHUS_TEST_EQUAL(passed, map_of_const_sgal3.timestamp().data(),
                      raw.data() + kQuat + 6, "");

    Eigen::Map<SGal3Type const> const_shallow_copy = map_of_const_sgal3;
    expectSame(passed, const_shallow_copy, map_of_const_sgal3, eps,
               "const shallow copy");

    // Contiguous data() access on a concrete SGal3 element.
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw.template head<4>();
    SGal3Type const const_sgal3(quat, raw.template segment<3>(kQuat).eval(),
                                raw.template segment<3>(kQuat + 3).eval(),
                                raw.template segment<1>(kQuat + 6).eval());
    for (int i = 0; i < SGal3Type::num_parameters; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_sgal3.data()[i], raw.data()[i], "");
    }

    // Round trip: fill a buffer from a group element, map it, and read back.
    Time t_orig;
    t_orig << Scalar(2.5);
    SGal3Type const G_original(
        SO3Type::exp(Vector3<Scalar>(Scalar(0.3), Scalar(0.2), Scalar(0.1))),
        Point(Scalar(1.5), Scalar(2.5), Scalar(3.5)),
        Point(Scalar(0.3), Scalar(0.4), Scalar(0.5)), t_orig);

    std::vector<Scalar> buffer(SGal3Type::num_parameters);
    toRawBuffer(G_original, buffer.data());

    Eigen::Map<SGal3Type> map_of_sgal3(buffer.data());
    SGal3Type const G_recovered(SO3Type(map_of_sgal3.rotation()),
                                Point(map_of_sgal3.translation()),
                                Point(map_of_sgal3.boost()),
                                Time(map_of_sgal3.timestamp()));
    expectSame(passed, G_recovered, G_original, eps, "Map round trip");

    Eigen::Map<SGal3Type> shallow_copy = map_of_sgal3;
    expectSame(passed, shallow_copy, map_of_sgal3, eps, "shallow copy");

    return passed;
  }

  bool testMutatingAccessors() {
    bool passed = true;
    Scalar const eps = Constants<Scalar>::epsilon();
    int const kQuat = SO3Type::num_parameters;

    // Modifications through an Eigen::Map must be reflected in the buffer.
    std::vector<Scalar> buffer(SGal3Type::num_parameters);
    toRawBuffer(SGal3Type(), buffer.data());

    Eigen::Map<SGal3Type> map_of_sgal3(buffer.data());

    Point const new_translation(Scalar(1), Scalar(2), Scalar(3));
    map_of_sgal3.translation() = new_translation;
    Point const new_boost(Scalar(0.1), Scalar(0.2), Scalar(0.3));
    map_of_sgal3.boost() = new_boost;

    for (int i = 0; i < 3; ++i) {
      SOPHUS_TEST_APPROX(passed, buffer[kQuat + i], new_translation(i), eps,
                         "map translation -> buffer component {}", i);
      SOPHUS_TEST_APPROX(passed, buffer[kQuat + 3 + i], new_boost(i), eps,
                         "map boost -> buffer component {}", i);
    }
    SOPHUS_TEST_APPROX(passed, map_of_sgal3.translation().eval(),
                       new_translation, eps, "");
    SOPHUS_TEST_APPROX(passed, map_of_sgal3.boost().eval(), new_boost, eps, "");

    // setQuaternion mutating accessor on a concrete element.
    SGal3Type se;
    SO3Type const R =
        SO3Type::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(0.0)));
    se.setQuaternion(R.unit_quaternion());
    SOPHUS_TEST_APPROX(passed, se.rotation().matrix(), R.matrix(), eps,
                       "setQuaternion");

    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Scalar const eps = Constants<Scalar>::epsilon();

    // Default constructor produces the identity element.
    SGal3Type const identity;
    SOPHUS_TEST_APPROX(passed, identity.rotation().matrix(),
                       Matrix3<Scalar>::Identity().eval(), eps,
                       "default ctor rotation");
    SOPHUS_TEST_APPROX(passed, identity.translation().eval(),
                       Point::Zero().eval(), eps, "default ctor translation");
    SOPHUS_TEST_APPROX(passed, identity.boost().eval(), Point::Zero().eval(),
                       eps, "default ctor boost");
    SOPHUS_TEST_APPROX(passed, identity.timestamp()(0, 0), Scalar(0), eps,
                       "default ctor timestamp");

    SO3Type const C =
        SO3Type::exp(Vector3<Scalar>(Scalar(0.2), Scalar(0.5), Scalar(0.0)));
    Point const p(Scalar(1), Scalar(-3), Scalar(0.5));
    Point const v(Scalar(0.3), Scalar(0.4), Scalar(0.5));
    Time t;
    t << Scalar(1.5);

    // Constructor from (SO3, translation, boost, timestamp).
    SGal3Type const G_from_so3(C, p, v, t);
    SOPHUS_TEST_APPROX(passed, G_from_so3.rotation().matrix(), C.matrix(), eps,
                       "SO3 ctor rotation");
    SOPHUS_TEST_APPROX(passed, G_from_so3.translation().eval(), p, eps,
                       "SO3 ctor translation");
    SOPHUS_TEST_APPROX(passed, G_from_so3.boost().eval(), v, eps,
                       "SO3 ctor boost");
    SOPHUS_TEST_APPROX(passed, G_from_so3.timestamp()(0, 0), t(0, 0), eps,
                       "SO3 ctor timestamp");

    // Constructor from (quaternion, translation, boost, timestamp).
    SGal3Type const G_from_quat(C.unit_quaternion(), p, v, t);
    expectSame(passed, G_from_quat, G_from_so3, eps, "quaternion ctor");

    // Copy constructor and copy assignment.
    SGal3Type const G_copy(G_from_so3);
    expectSame(passed, G_copy, G_from_so3, eps, "copy ctor");

    SGal3Type G_assigned;
    G_assigned = G_from_so3;
    expectSame(passed, G_assigned, G_from_so3, eps, "copy assignment");

    return passed;
  }

  std::vector<SGal3Type, Eigen::aligned_allocator<SGal3Type>> sgal3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
};

int test_sgal3() {
  using std::cerr;
  using std::endl;

  cerr << "Test SGal3" << endl << endl;
  cerr << "Double tests: " << endl;
  Tests<double>().runAll();

//   cerr << "Float tests: " << endl;
//   Tests<float>().runAll();
//   #if SOPHUS_CERES
//     cerr << "ceres::Jet<double, 3> tests: " << endl;
//     Tests<ceres::Jet<double, 3>>().runAll();
//   #endif

  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_sgal3(); }