#include <algorithm>
#include <iostream>
#include <vector>

#include <sophus/arist3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.

namespace Eigen {
template class Map<Sophus::Arist3<double>>;
template class Map<Sophus::Arist3<double> const>;
}  // namespace Eigen

namespace Sophus {

template class Arist3<double>;
template class Arist3<float>;
#if SOPHUS_CERES
template class Arist3<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
class Tests {
 public:
  using Arist3Type = Arist3<Scalar>;
  using SO3Type = SO3<Scalar>;
  using Point = Vector3<Scalar>;
  using Tangent = typename Arist3<Scalar>::TangentVector;
  using Event = typename Arist3<Scalar>::EventType;  // [x (3), tau (1)]
  // Arist3 carries a scalar time-stamp, represented as a 1x1 vector.
  using Time = Eigen::Matrix<Scalar, 1, 1>;
  Scalar const kPi = Constants<Scalar>::pi();

  Tests() {
    arist3_vec_ = getTestArist3s();

    // Tangent layout: [rho (translation, 3), theta (rotation, 3),
    //                  iota (time, 1)].  (No boost component, unlike SGal3.)
    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0),  // rho
        Scalar(0), Scalar(0), Scalar(0),     // theta
        Scalar(0);                           // iota
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
        Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
        Scalar(0.5);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(-5), Scalar(10), Scalar(0.1), Scalar(0.2),
        Scalar(0.3), Scalar(1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1),
        Scalar(0.2);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(2), Scalar(-1), Scalar(0), Scalar(0.1), Scalar(0.05),
        Scalar(0.15), Scalar(0.7);
    tangent_vec_.push_back(tmp);
  }

  void runAll() {
    bool passed = testGroupOperations();
    passed &= testExpLog();
    passed &= testExpLogRandom();
    passed &= testMatrixAction();
    passed &= testRawDataAcces();
    passed &= testMutatingAccessors();
    passed &= testConstructors();
    processTestResult(passed);
  }

 private:
  // Builds a set of representative Arist3 elements to exercise the tests.
  std::vector<Arist3Type, Eigen::aligned_allocator<Arist3Type>>
  getTestArist3s() {
    std::vector<Arist3Type, Eigen::aligned_allocator<Arist3Type>> arist3_vec;

    SO3Type const C0;  // identity rotation
    SO3Type const C1 =
        SO3Type::exp(Vector3<Scalar>(Scalar(0.2), Scalar(0.5), Scalar(0.1)));
    SO3Type const C2 = SO3Type::exp(
        Vector3<Scalar>(Scalar(0), Scalar(0), Scalar(kPi) / Scalar(2)));

    Point const p0 = Point::Zero();
    Point const p1(Scalar(1), Scalar(2), Scalar(3));
    Time const t0 = Time::Zero();
    Time t1;
    t1 << Scalar(1.5);

    arist3_vec.push_back(Arist3Type());             // identity
    arist3_vec.push_back(Arist3Type(C1, p0, t0));   // rotation only
    arist3_vec.push_back(Arist3Type(C0, p1, t0));   // translation only
    arist3_vec.push_back(Arist3Type(C0, p0, t1));   // time only
    arist3_vec.push_back(Arist3Type(C2, p1, t1));   // general element
    arist3_vec.push_back(Arist3Type(C1, -p1, t1));  // general element

    return arist3_vec;
  }

  // Copies a group element into a contiguous raw parameter buffer using the
  // ``[quaternion, translation, timestamp]`` storage order.
  template <class Derived>
  static void toRawBuffer(Arist3Base<Derived> const& G, Scalar* buffer) {
    Eigen::Map<SO3Type>(buffer).setQuaternion(G.rotation().unit_quaternion());
    std::copy(G.translation().data(), G.translation().data() + 3,
              buffer + SO3Type::num_parameters);
    std::copy(G.timestamp().data(), G.timestamp().data() + 1,
              buffer + SO3Type::num_parameters + 3);
  }

  // Component-wise approximate comparison of two Arist3 elements. Templated on
  // the CRTP-derived types so that it works for Arist3<Scalar> as well as for
  // Eigen::Map<Arist3<Scalar>> and Eigen::Map<Arist3<Scalar> const>.
  template <class LhsDerived, class RhsDerived>
  void expectSame(bool& passed, Arist3Base<LhsDerived> const& lhs,
                  Arist3Base<RhsDerived> const& rhs, Scalar prec,
                  char const* msg) {
    SOPHUS_TEST_APPROX(passed, lhs.rotation().matrix(), rhs.rotation().matrix(),
                       prec, "{} (rotation)", msg);
    SOPHUS_TEST_APPROX(passed, lhs.translation().eval(),
                       rhs.translation().eval(), prec, "{} (translation)", msg);
    SOPHUS_TEST_APPROX(passed, lhs.timestamp()(0, 0), rhs.timestamp()(0, 0),
                       prec, "{} (timestamp)", msg);
  }

  // G * G.inverse() == identity, and G * identity == identity * G == G.
  bool testGroupOperations() {
    bool passed = true;
    Scalar const eps = Constants<Scalar>::epsilon();
    Arist3Type const identity;

    for (size_t i = 0; i < arist3_vec_.size(); ++i) {
      Arist3Type const& G = arist3_vec_[i];

      Arist3Type const should_be_identity = G * G.inverse();
      SOPHUS_TEST_APPROX(passed, should_be_identity.rotation().matrix(),
                         Matrix3<Scalar>::Identity().eval(), eps,
                         "G * G.inverse() rotation case: {}", i);
      SOPHUS_TEST_APPROX(passed, should_be_identity.translation().eval(),
                         Point::Zero().eval(), eps,
                         "G * G.inverse() translation case: {}", i);
      SOPHUS_TEST_APPROX(passed, should_be_identity.timestamp()(0, 0),
                         Scalar(0), eps, "G * G.inverse() timestamp case: {}",
                         i);

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
      Tangent const g_recovered = Arist3Type::exp(g).log();
      SOPHUS_TEST_APPROX(passed, g_recovered, g, prec,
                         "g - log(exp(g)) case: {}", i);
    }

    // Round trip starting from the group: G - exp(log(G)).
    for (size_t i = 0; i < arist3_vec_.size(); ++i) {
      Arist3Type const& G = arist3_vec_[i];
      Arist3Type const G_recovered = Arist3Type::exp(G.log());
      expectSame(passed, G_recovered, G, prec, "G - exp(log(G))");
    }

    // Small perturbations (important for iterative optimization).
    Tangent small;
    small << Scalar(1e-6), Scalar(2e-6), Scalar(3e-6),  // rho
        Scalar(1e-7), Scalar(2e-7), Scalar(3e-7),       // theta
        Scalar(5e-7);                                   // iota
    Tangent const small_recovered = Arist3Type::exp(small).log();
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
        g(6) = mag * Scalar(0.5) + Scalar(0.1);  // non-zero time component
        Tangent const g_recovered = Arist3Type::exp(g).log();
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

  // The point/event action must agree with the 5x5 matrix representation, and
  // matrix() must be a group homomorphism. This directly guards the time
  // component update in operator*(EventType) -- i.e. the tau -> tau + t term
  // that was previously being dropped.
  bool testMatrixAction() {
    bool passed = true;
    Scalar const prec = Constants<Scalar>::epsilonSqrt();
    using Vector5 = Eigen::Matrix<Scalar, 5, 1>;

    std::vector<Event, Eigen::aligned_allocator<Event>> events;
    {
      Event e;
      e << Scalar(0), Scalar(0), Scalar(0), Scalar(0);
      events.push_back(e);
    }
    {
      Event e;
      e << Scalar(1), Scalar(2), Scalar(3), Scalar(0);
      events.push_back(e);
    }
    {
      Event e;
      e << Scalar(-2), Scalar(0.5), Scalar(4), Scalar(2.5);
      events.push_back(e);
    }
    {
      Event e;
      e << Scalar(1), Scalar(-1), Scalar(1), Scalar(-1.5);
      events.push_back(e);
    }

    // Event action via operator* must match the 5x5 matrix acting on the
    // homogeneous event [x, tau, 1].
    for (size_t i = 0; i < arist3_vec_.size(); ++i) {
      Arist3Type const& G = arist3_vec_[i];
      for (size_t j = 0; j < events.size(); ++j) {
        Event const& e = events[j];

        Event const mapped = G * e;

        Vector5 hom;
        hom.template head<3>() = e.template head<3>();
        hom(3) = e(3);
        hom(4) = Scalar(1);
        Vector5 const hom_mapped = G.matrix() * hom;

        SOPHUS_TEST_APPROX(passed, mapped.template head<3>().eval(),
                           hom_mapped.template head<3>().eval(), prec,
                           "event action vs matrix (space) G {}, event {}", i,
                           j);
        SOPHUS_TEST_APPROX(passed, mapped(3), hom_mapped(3), prec,
                           "event action vs matrix (time) G {}, event {}", i, j);
      }
    }

    // matrix() is a homomorphism: (A * B).matrix() == A.matrix() * B.matrix().
    for (size_t i = 0; i < arist3_vec_.size(); ++i) {
      for (size_t j = 0; j < arist3_vec_.size(); ++j) {
        Arist3Type const& A = arist3_vec_[i];
        Arist3Type const& B = arist3_vec_[j];
        SOPHUS_TEST_APPROX(passed, (A * B).matrix(),
                           (A.matrix() * B.matrix()).eval(), prec,
                           "matrix homomorphism A {}, B {}", i, j);
      }
    }
    return passed;
  }

  bool testRawDataAcces() {
    bool passed = true;
    Scalar const eps = Constants<Scalar>::epsilon();
    int const kQuat = SO3Type::num_parameters;  // == 4

    // Raw parameter layout: [quaternion (4), translation (3), timestamp (1)]
    //                       == num_parameters (== 8).
    Eigen::Matrix<Scalar, Arist3Type::num_parameters, 1> raw;
    raw << Scalar(0), Scalar(0), Scalar(0), Scalar(1),  // identity quaternion
        Scalar(1), Scalar(2), Scalar(3),                // translation
        Scalar(1.5);                                    // timestamp

    Eigen::Map<Arist3Type const> map_of_const_arist3(raw.data());
    SOPHUS_TEST_APPROX(
        passed,
        map_of_const_arist3.rotation().unit_quaternion().coeffs().eval(),
        raw.template head<4>().eval(), eps, "");
    SOPHUS_TEST_APPROX(passed, map_of_const_arist3.translation().eval(),
                       raw.template segment<3>(kQuat).eval(), eps, "");
    SOPHUS_TEST_APPROX(passed, map_of_const_arist3.timestamp()(0, 0),
                       raw(kQuat + 3), eps, "");

    SOPHUS_TEST_EQUAL(
        passed,
        map_of_const_arist3.rotation().unit_quaternion().coeffs().data(),
        raw.data(), "");
    SOPHUS_TEST_EQUAL(passed, map_of_const_arist3.translation().data(),
                      raw.data() + kQuat, "");
    SOPHUS_TEST_EQUAL(passed, map_of_const_arist3.timestamp().data(),
                      raw.data() + kQuat + 3, "");

    Eigen::Map<Arist3Type const> const_shallow_copy = map_of_const_arist3;
    expectSame(passed, const_shallow_copy, map_of_const_arist3, eps,
               "const shallow copy");

    // Contiguous data() access on a concrete Arist3 element.
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw.template head<4>();
    Arist3Type const const_arist3(quat, raw.template segment<3>(kQuat).eval(),
                                  raw.template segment<1>(kQuat + 3).eval());
    for (int i = 0; i < Arist3Type::num_parameters; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_arist3.data()[i], raw.data()[i], "");
    }

    // Round trip: fill a buffer from a group element, map it, and read back.
    Time t_orig;
    t_orig << Scalar(2.5);
    Arist3Type const G_original(
        SO3Type::exp(Vector3<Scalar>(Scalar(0.3), Scalar(0.2), Scalar(0.1))),
        Point(Scalar(1.5), Scalar(2.5), Scalar(3.5)), t_orig);

    std::vector<Scalar> buffer(Arist3Type::num_parameters);
    toRawBuffer(G_original, buffer.data());

    Eigen::Map<Arist3Type> map_of_arist3(buffer.data());
    Arist3Type const G_recovered(SO3Type(map_of_arist3.rotation()),
                                 Point(map_of_arist3.translation()),
                                 Time(map_of_arist3.timestamp()));
    expectSame(passed, G_recovered, G_original, eps, "Map round trip");

    Eigen::Map<Arist3Type> shallow_copy = map_of_arist3;
    expectSame(passed, shallow_copy, map_of_arist3, eps, "shallow copy");

    return passed;
  }

  bool testMutatingAccessors() {
    bool passed = true;
    Scalar const eps = Constants<Scalar>::epsilon();
    int const kQuat = SO3Type::num_parameters;

    // Modifications through an Eigen::Map must be reflected in the buffer.
    std::vector<Scalar> buffer(Arist3Type::num_parameters);
    toRawBuffer(Arist3Type(), buffer.data());

    Eigen::Map<Arist3Type> map_of_arist3(buffer.data());

    Point const new_translation(Scalar(1), Scalar(2), Scalar(3));
    map_of_arist3.translation() = new_translation;
    Time new_time;
    new_time << Scalar(4.2);
    map_of_arist3.timestamp() = new_time;

    for (int i = 0; i < 3; ++i) {
      SOPHUS_TEST_APPROX(passed, buffer[kQuat + i], new_translation(i), eps,
                         "map translation -> buffer component {}", i);
    }
    SOPHUS_TEST_APPROX(passed, buffer[kQuat + 3], new_time(0, 0), eps,
                       "map timestamp -> buffer");
    SOPHUS_TEST_APPROX(passed, map_of_arist3.translation().eval(),
                       new_translation, eps, "");
    SOPHUS_TEST_APPROX(passed, map_of_arist3.timestamp()(0, 0), new_time(0, 0),
                       eps, "");

    // setQuaternion mutating accessor on a concrete element.
    Arist3Type se;
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
    Arist3Type const identity;
    SOPHUS_TEST_APPROX(passed, identity.rotation().matrix(),
                       Matrix3<Scalar>::Identity().eval(), eps,
                       "default ctor rotation");
    SOPHUS_TEST_APPROX(passed, identity.translation().eval(),
                       Point::Zero().eval(), eps, "default ctor translation");
    SOPHUS_TEST_APPROX(passed, identity.timestamp()(0, 0), Scalar(0), eps,
                       "default ctor timestamp");

    SO3Type const C =
        SO3Type::exp(Vector3<Scalar>(Scalar(0.2), Scalar(0.5), Scalar(0.0)));
    Point const p(Scalar(1), Scalar(-3), Scalar(0.5));
    Time t;
    t << Scalar(1.5);

    // Constructor from (SO3, translation, timestamp).
    Arist3Type const G_from_so3(C, p, t);
    SOPHUS_TEST_APPROX(passed, G_from_so3.rotation().matrix(), C.matrix(), eps,
                       "SO3 ctor rotation");
    SOPHUS_TEST_APPROX(passed, G_from_so3.translation().eval(), p, eps,
                       "SO3 ctor translation");
    SOPHUS_TEST_APPROX(passed, G_from_so3.timestamp()(0, 0), t(0, 0), eps,
                       "SO3 ctor timestamp");

    // Constructor from (quaternion, translation, timestamp).
    Arist3Type const G_from_quat(C.unit_quaternion(), p, t);
    expectSame(passed, G_from_quat, G_from_so3, eps, "quaternion ctor");

    // Copy constructor and copy assignment.
    Arist3Type const G_copy(G_from_so3);
    expectSame(passed, G_copy, G_from_so3, eps, "copy ctor");

    Arist3Type G_assigned;
    G_assigned = G_from_so3;
    expectSame(passed, G_assigned, G_from_so3, eps, "copy assignment");

    return passed;
  }

  std::vector<Arist3Type, Eigen::aligned_allocator<Arist3Type>> arist3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
};

int test_arist3() {
  using std::cerr;
  using std::endl;

  cerr << "Test Arist3" << endl << endl;
  cerr << "Double tests: " << endl;
  Tests<double>().runAll();

  cerr << "Float tests: " << endl;
  Tests<float>().runAll();
#if SOPHUS_CERES
  cerr << "ceres::Jet<double, 3> tests: " << endl;
  Tests<ceres::Jet<double, 3>>().runAll();
#endif

  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_arist3(); }