/* ----------------------------------------------------------------------------

 * Author: Abhishek Goudar.

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

#pragma once

#include <sophus/se3.hpp>

namespace Sophus
{
    template<typename Scalar>
    using Vector1 = Matrix<Scalar, 1, 1>;
    using Vector1d = Vector1<double>;
    using Vector1f = Vector1<float>;
    // /
    template<typename Scalar, int Options = 0>
    class Arist3;
    using Arist3d = Arist3<double>;
    using Arist3f = Arist3<float>;
}

namespace Eigen
{
    namespace internal
    {
        template<typename Scalar_, int Options_>
        struct traits<Sophus::Arist3<Scalar_, Options_>>
        {
            static constexpr int Options = Options_;
            using Scalar = Scalar_;
            using RotationType = Sophus::SO3<Scalar_, Options_>;
            using TranslationType = Eigen::Matrix<Scalar_, 3, 1, Options_>;
            using TimestampType =  Eigen::Matrix<Scalar_, 1, 1, Options_>;
        };
        //
        template<typename Scalar_, int Options_>
        struct traits<Map<Sophus::Arist3<Scalar_>, Options_>>
        {
            static constexpr int Options = Options_;
            using Scalar = Scalar_;
            using RotationType = Map<Sophus::SO3<Scalar_>, Options_>;
            using TranslationType = Map<Eigen::Matrix<Scalar_, 3, 1>, Options_>;
            using TimestampType =  Map<Eigen::Matrix<Scalar_, 1, 1>, Options_>;
        };
        //
        template<typename Scalar_, int Options_>
        struct traits<Map<Sophus::Arist3<Scalar_> const, Options_>>
        {
            static constexpr int Options = Options_;
            using Scalar = Scalar_;
            using RotationType = Map<Sophus::SO3<Scalar_> const, Options_>;
            using TranslationType = Map<Eigen::Matrix<Scalar_, 3, 1> const, Options_>;
            using TimestampType =  Map<Eigen::Matrix<Scalar_, 1, 1> const, Options_>;
        };
    }
    //
}

namespace Sophus
{
    /**
     * @brief arist3 tangent vector = [rho, theta, iota]
     * where rho - translational component,
     * theta - angular component
     * iota - temporal component
     */
    namespace arist3{
        /**
         * 
         */
        template<typename Scalar>
        static inline const Eigen::Matrix<Scalar, 3, 1> rho(
            const typename Arist3<Scalar>::TangentVector& a)
        {
            return a.template segment<3>(0);
        }
        /**
         * 
         */
        template<typename Scalar>
        static inline const Eigen::Matrix<Scalar, 3, 1> theta(
            const typename Arist3<Scalar>::TangentVector& a)
        {
            return a.template segment<3>(3);
        }
        /**
         * 
         */
        template<typename Scalar>
        static inline const Eigen::Matrix<Scalar, 1, 1> iota(
            const typename Arist3<Scalar>::TangentVector& a)
        {
            return a.template segment<1>(6);
        }
        /**
         * 
         */
        static inline std::vector<int> get_tangent_dims_translation()
        {
            return {0,1,2};
        }
        //
        static inline std::vector<int> get_tangent_dims_rotation()
        {
            return {3,4,5};
        }
        //
        static inline std::vector<int> get_tangent_dims_time()
        {
            return {6};
        }
    };


    template<typename Derived>
    class Arist3Base{
        //
        public:
        //
        using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
        using RotationType = typename Eigen::internal::traits<Derived>::RotationType;
        using TranslationType = typename Eigen::internal::traits<Derived>::TranslationType;
        using TimestampType = typename Eigen::internal::traits<Derived>::TimestampType;
        using EventType = Vector<Scalar, 4>;
        //
        static int constexpr MATRIX_SIZE = 5;
        //
        static constexpr uint8_t DoF = 7;
        //
        /**
         * @brief non-mutable accessor of translation vector
         */
        TranslationType const& translation() const {
            return static_cast<Derived const*>(this)->translation();
        }
        /**
         * @brief mutable accessor of translation vector
         */
        TranslationType& translation(){
            return static_cast<Derived*>(this)->translation();
        }
        /**
         * @brief non-mutable accessor of rotation vector
         */
        RotationType const& rotation() const {
            return static_cast<Derived const*>(this)->rotation();
        }
        /**
         * @brief mutable accessor of rotation vector
         */
        RotationType& rotation(){
            return static_cast<Derived*>(this)->rotation();
        }

        /**
         * @brief non-mutable accessor of timestamp vector
         */
        TimestampType const& timestamp() const {
            return static_cast<Derived const*>(this)->timestamp();
        }
        /**
         * @brief mutable accessor of timestamp vector
         */
        TimestampType& timestamp(){
            return static_cast<Derived*>(this)->timestamp();
        }
        /**
         * @brief accessor for pose
         */
        Sophus::SE3<Scalar> const pose() const{
            return Sophus::SE3<Scalar>(
                rotation(),
                translation());
        }
        /**
         * @brief returns Arist3 element as matrix
         * [C, 0, r]
         * [0, 1, t]
         * [0, 0, 1]
         * where C is rotation matrix, 
         * r is translation
         * t is timestamp
         * @return Arist3 element as 5x5 matrix
         */
        Eigen::Matrix<Scalar, MATRIX_SIZE, MATRIX_SIZE> matrix() const 
        {
            Eigen::Matrix<Scalar, MATRIX_SIZE, MATRIX_SIZE> A_;
            A_.setIdentity();
            A_.template block<3,3>(0,0) = rotation().matrix();
            A_.template block<3,1>(0,4) = translation();
            A_.template block<1,1>(3,4) = timestamp();
            return A_;
        }

        /**
         * @brief
         */
        template<typename OtherDerived>
        Arist3Base<Derived>& operator=(Arist3Base<OtherDerived> const& other) {
            rotation() = other.rotation();
            translation() = other.translation();
            timestamp() = other.timestamp();
            return *this;
        }
        /**
         * 
         */
        EventType operator*(EventType const& p) const {
            EventType pA;
            pA.template head<3>() = rotation() * p.template head<3>()
                 + translation();
            pA.template tail<1>() = timestamp() + p.template tail<1>();
            return pA;
        }
        /**
         * @brief Group composition (multiplication)
         * Computes the product of two Arist(3) elements following the group operation:
         * (C1, r1, t1) * (C2, r2, t2) = 
	     * (C1*C2, C1*r2 + r1, t1 + t2)
         * TODO: (abhi) Need to resolve this. The return type is not 
         * storage agnostic
         */
        template<typename OtherDerived>
        Arist3<Scalar> operator*(const Arist3Base<OtherDerived>& other) const {
            return Arist3<Scalar>(
                rotation() * other.rotation(),
                rotation() * other.translation() + translation(),
                timestamp() + other.timestamp()
            );
        }
        /**
         * 
         */
        Arist3<Scalar> inverse() const
        {
            const Sophus::SO3<Scalar> C_inv = rotation().inverse();
            return Arist3<Scalar>(
                C_inv,
                C_inv * (-1. * translation()),
                -timestamp()
            );
        }
    };
    //
    template<typename Scalar, int Options>
    class Arist3 : public Arist3Base<Arist3<Scalar, Options>>
    {
        public:
        //
        using Base = Arist3Base<Arist3<Scalar, Options>>;
        using RotationType = typename Base::RotationType;
        using TranslationType = typename Base::TranslationType;
        using TimestampType = typename Base::TimestampType;
        //
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        //
        static constexpr uint8_t num_parameters = 8;
        //
        static constexpr uint8_t DoF = Base::DoF;
        //
        using TangentVector = Eigen::Matrix<Scalar, DoF, 1>;
        /**
         * 
         */
        // TODO: (abhi): its odd the other constructor doesn't do this
        Arist3() : 
            rotation_(Sophus::SO3<Scalar>()),
            translation_(TranslationType::Zero()),
            timestamp_(TimestampType::Zero()) {
            static_assert(std::is_standard_layout<Arist3>::value,
                            "Assume standard layout for the use of offsetof check below.");
            static_assert(
                offsetof(Arist3, rotation_) + sizeof(Scalar) * SO3<Scalar>::num_parameters ==
                    offsetof(Arist3, translation_),
                "This class assumes packed storage, in particular"
                "when using [this->data(), this-data() + "
                "num_parameters] to access the raw data in a contiguous fashion.");
            static_assert(
                offsetof(Arist3, translation_) + sizeof(Scalar) * 3 ==
                    offsetof(Arist3, timestamp_),
                "This class assumes packed storage, in particular"
                "when using [this->data(), this-data() + "
                "num_parameters] to access the raw data in a contiguous fashion.");
        }
        /**
         * 
         */
        Arist3(const Eigen::Quaternion<Scalar>& q_,
            const Eigen::Matrix<Scalar, 3, 1> p_,
            const Eigen::Matrix<Scalar, 1, 1> t_) : 
            rotation_(q_),
            translation_(p_),
            timestamp_(t_){};
        /**
         * @brief
         */
        Arist3(const Sophus::SO3<Scalar>& C_,
            const Eigen::Matrix<Scalar, 3, 1> p_,
            const Eigen::Matrix<Scalar, 1, 1> t_) : 
            rotation_(C_),
            translation_(p_),
            timestamp_(t_){};
        /**
         * @brief
         */
        void setIdentity()
        {
            // this assumes there is a Eigen::Quaternion
            // compatible constructor
            rotation_ = RotationType();
            translation_.setZero();
            timestamp_ = Vector1<Scalar>(Scalar(0.));
        }
        /**
         * 
         */
        Scalar* data() {
            return rotation_.data();
        }
        /**
         * 
         */
        Scalar const* data() const {
            return rotation_.data();
        }
        /**
         * @brief non-mutable accessor of translation vector
         */
        TranslationType const& translation() const {
            return translation_;
        }
        /**
         * @brief mutable accessor of translation vector
         */
        TranslationType& translation(){
            return translation_;
        }
        /**
         * @brief non-mutable accessor of rotation vector
         */
        RotationType const& rotation() const {
            return rotation_;
        }
        /**
         * @brief mutable accessor of rotation vector
         */
        RotationType& rotation(){
            return rotation_;
        }
        /**
         * @brief set quaternion
         */
        SOPHUS_FUNC void setQuaternion(Eigen::Quaternion<Scalar> const& quat) {
            rotation().setQuaternion(quat);
        }
        /**
         * @brief non-mutable accessor of timestamp vector
         */
        TimestampType const& timestamp() const {
            return timestamp_;
        }
        /**
         * @brief mutable accessor of timestamp vector
         */
        TimestampType& timestamp(){
            return timestamp_;
        }
        /**
         * @brief Closed-form exponential map for Arist(3)
         * Maps from the Lie algebra Arist(3) (tangent space) to the Lie group Arist(3)
         * 
         * @param g_ Tangent vector in arist(3): [rho, theta, iota]
         * @return Arist(3) group element
         */
        static Arist3 exp(const TangentVector& a_)
        {
            Eigen::Matrix<Scalar, 3, 3> D = 
                Sophus::SO3<Scalar>::leftJacobian(arist3::theta<Scalar>(a_));
            //
            // Exponential map formula:
            // C = exp(theta^)
            // r = D*rho
            // t = iota
            return Arist3(
                Sophus::SO3<Scalar>::exp(arist3::theta<Scalar>(a_)),
                D * arist3::rho<Scalar>(a_),
                arist3::iota<Scalar>(a_));
        }
        /**
         * @brief Closed-form logarithm map for the Arist(3)
         * Maps from the Lie group Arist(3) to its Lie algebra Arist(3) (tangent space)
         * 
         * 
         * @param G Arist(3) group element
         * @return Tangent vector in arist(3): [rho, theta, iota]
         */
        TangentVector log() const
        {
            TangentVector a;
            ///
            // Extract rotation logarithm
            const Eigen::Vector<Scalar, 3> theta = this->rotation().log();
            //
            // // Compute inverses of left Jacobian inverse 
            const Eigen::Matrix<Scalar, 3, 3> D_inv = Sophus::SO3<Scalar>::leftJacobianInverse(theta);
            //
            // Logarithmic map formula:
            // rho = D_inv * r
            // theta = log(C)^v
            // iota = t
            a.template segment<3>(0) = D_inv * this->translation();
            a.template segment<3>(3) = theta;
            a.template segment<1>(6) = this->timestamp();
            //
            return a;
        }
        //
        protected:
        //
        RotationType rotation_;
        TranslationType translation_;
        TimestampType timestamp_;
    };
    //
} // namespace Sophus

namespace Eigen
{
    template<typename Scalar, int Options>
    class Map<Sophus::Arist3<Scalar>, Options> : 
        public Sophus::Arist3Base<Map<Sophus::Arist3<Scalar>, Options>>
    {
        using Base = Sophus::Arist3Base<Map<Sophus::Arist3<Scalar>, Options>>;
        using RotationType = typename Base::RotationType;
        using TranslationType = typename Base::TranslationType;
        using TimestampType = typename Base::TimestampType;
        //
        public:
        //
        Map(Scalar* data) :
            rotation_(data),
            // TODO: get num_parameters from RotationType?
            translation_(data + Sophus::SO3<Scalar>::num_parameters),
            timestamp_(data + Sophus::SO3<Scalar>::num_parameters + 3)
        {
        };
        //
        /**
         * @brief non-mutable accessor of translation vector
         */
        TranslationType const& translation() const {
            return translation_;
        }
        /**
         * @brief mutable accessor of translation vector
         */
        TranslationType& translation(){
            return translation_;
        }
        /**
         * @brief non-mutable accessor of rotation vector
         */
        RotationType const& rotation() const {
            return rotation_;
        }
        /**
         * @brief mutable accessor of rotation vector
         */
        RotationType& rotation(){
            return rotation_;
        }

        /**
         * @brief non-mutable accessor of timestamp vector
         */
        TimestampType const& timestamp() const {
            return timestamp_;
        }
        /**
         * @brief mutable accessor of timestamp vector
         */
        TimestampType& timestamp(){
            return timestamp_;
        }
        // This overrides the default assignment operator
        using Base::operator=;
        //
        protected:
        //
        RotationType rotation_;
        TranslationType translation_;
        TimestampType timestamp_;
    };
    /**
     * 
     */
    template<typename Scalar, int Options>
    class Map<Sophus::Arist3<Scalar> const, Options> : 
        public Sophus::Arist3Base<Map<Sophus::Arist3<Scalar> const, Options>>
    {
        public:
        //
        using Base = typename Sophus::Arist3Base<Map<Sophus::Arist3<Scalar> const, Options>>;
        using RotationType = typename Base::RotationType;
        using TranslationType = typename Base::TranslationType;
        using TimestampType = typename Base::TimestampType;
        // Const maps are read-only views — disable assignment.
        template<typename OtherDerived>
        Map& operator=(Sophus::Arist3Base<OtherDerived> const&) = delete;
        Map& operator=(Map const&) = delete;
        //
        Map(Scalar const* data) :
            rotation_(data),
            // TODO: get num_parameters from RotationType?
            translation_(data + Sophus::SO3<Scalar>::num_parameters),
            timestamp_(data + Sophus::SO3<Scalar>::num_parameters + 3)
        {
        };
        //
        /**
         * @brief non-mutable accessor of translation vector
         */
        TranslationType const& translation() const {
            return translation_;
        }
        /**
         * @brief non-mutable accessor of rotation vector
         */
        RotationType const& rotation() const {
            return rotation_;
        }
        /**
         * @brief non-mutable accessor of timestamp vector
         */
        TimestampType const& timestamp() const {
            return timestamp_;
        }
        // This overrides the default assignment operator
        using Base::operator=;
        //
        protected:
        //
        RotationType rotation_;
        TranslationType translation_;
        TimestampType timestamp_;
    };
}
