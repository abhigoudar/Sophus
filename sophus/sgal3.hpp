/* ----------------------------------------------------------------------------

 * Author: Abhishek Goudar.

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

#pragma once

#include <sophus/so3.hpp>

namespace Sophus
{
    template<typename Scalar>
    using Vector1 = Matrix<Scalar, 1, 1>;
    using Vector1d = Vector1<double>;
    using Vector1f = Vector1<float>;
    // /
    template<typename Scalar, int Options = 0>
    class SGal3;
    using SGal3d = SGal3<double>;
    using SGal3f = SGal3<float>;
}

namespace Eigen
{
    namespace internal
    {
        template<typename Scalar_, int Options_>
        struct traits<Sophus::SGal3<Scalar_, Options_>>
        {
            static constexpr int Options = Options_;
            using Scalar = Scalar_;
            using RotationType = Sophus::SO3<Scalar_, Options_>;
            using TranslationType = Eigen::Matrix<Scalar_, 3, 1, Options_>;
            using BoostType = Eigen::Matrix<Scalar_, 3, 1, Options_>;
            using TimestampType =  Eigen::Matrix<Scalar_, 1, 1, Options_>;
        };
        //
        template<typename Scalar_, int Options_>
        struct traits<Map<Sophus::SGal3<Scalar_>, Options_>>
        {
            static constexpr int Options = Options_;
            using Scalar = Scalar_;
            using RotationType = Map<Sophus::SO3<Scalar_>, Options_>;
            using TranslationType = Map<Eigen::Matrix<Scalar_, 3, 1>, Options_>;
            using BoostType = Map<Eigen::Matrix<Scalar_, 3, 1>, Options_>;
            using TimestampType =  Map<Eigen::Matrix<Scalar_, 1, 1>, Options_>;
        };
        //
        template<typename Scalar_, int Options_>
        struct traits<Map<Sophus::SGal3<Scalar_> const, Options_>>
        {
            static constexpr int Options = Options_;
            using Scalar = Scalar_;
            using RotationType = Map<Sophus::SO3<Scalar_> const, Options_>;
            using TranslationType = Map<Eigen::Matrix<Scalar_, 3, 1> const, Options_>;
            using BoostType = Map<Eigen::Matrix<Scalar_, 3, 1> const, Options_>;
            using TimestampType =  Map<Eigen::Matrix<Scalar_, 1, 1> const, Options_>;
        };
    }
    //
}

namespace Sophus
{
    /**
     * @brief sgal3 tangent vector = [rho, nu, theta, iota]
     * where rho - translational component,
     * nu - velocity component,
     * theta - angular component
     * iota - temporal component
     */
    namespace sgal3{
        /**
         * 
         */
        template<typename Scalar>
        static inline const Eigen::Matrix<Scalar, 3, 1> rho(
            const typename SGal3<Scalar>::TangentVector& g)
        {
            return g.template segment<3>(0);
        }
        /**
         * 
         */
        template<typename Scalar>
        static inline const Eigen::Matrix<Scalar, 3, 1> nu(
            const typename SGal3<Scalar>::TangentVector& g)
        {
            return g.template segment<3>(3);
        }
        /**
         * 
         */
        template<typename Scalar>
        static inline const Eigen::Matrix<Scalar, 3, 1> theta(
            const typename SGal3<Scalar>::TangentVector& g)
        {
            return g.template segment<3>(6);
        }
        /**
         * 
         */
        template<typename Scalar>
        static inline const Eigen::Matrix<Scalar, 1, 1> iota(
            const typename SGal3<Scalar>::TangentVector& g)
        {
            return g.template segment<1>(9);
        }
    };


    template<typename Derived>
    class SGal3Base{
        //
        public:
        //
        using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
        using RotationType = typename Eigen::internal::traits<Derived>::RotationType;
        using TranslationType = typename Eigen::internal::traits<Derived>::TranslationType;
        using BoostType = typename Eigen::internal::traits<Derived>::BoostType;
        using TimestampType = typename Eigen::internal::traits<Derived>::TimestampType;
        using EventType = Vector<Scalar, 4>;
        //
        static int constexpr MATRIX_SIZE = 5;
        //
        static constexpr uint8_t DoF = 10;
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
         * @brief non-mutable accessor of boost vector
         */
        BoostType const& boost() const {
            return static_cast<Derived const*>(this)->boost();
        }
        /**
         * @brief mutable accessor of boost vector
         */
        BoostType& boost(){
            return static_cast<Derived*>(this)->boost();
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
         * @brief returns SGal3 element as matrix
         * [C, v, r]
         * [0, 1, t]
         * [0, 0, 1]
         * where C is rotation matrix, 
         * v is velocity(boost),
         * r is translation
         * t is timestamp
         * @return SGal3 element as 5x5 matrix
         */
        Eigen::Matrix<Scalar, MATRIX_SIZE, MATRIX_SIZE> matrix()
        {
            Eigen::Matrix<Scalar, MATRIX_SIZE, MATRIX_SIZE> G_;
            G_.setIdentity();
            G_.template block<3,3>(0,0) = rotation().matrix();
            G_.template block<3,1>(0,3) = boost();
            G_.template block<3,1>(0,4) = translation();
            G_.template block<1,1>(3,4) = timestamp();
            return G_;
        }

        /**
         * @brief
         */
        template<typename OtherDerived>
        SGal3Base<Derived>& operator=(SGal3Base<OtherDerived> const& other) {
            rotation() = other.rotation();
            translation() = other.translation();
            boost() = other.boost();
            timestamp() = other.timestamp();
            return *this;
        }
        /**
         * 
         */
        EventType operator*(EventType const& p) const {
            EventType pG;
            pG.template head<3>() = rotation() * p.template head<3>() + 
                boost() * p.template tail<1>() + translation();
            pG.template tail<1>() = p.template tail<1>();
            return pG;
        }
        /**
         * @brief Group composition (multiplication)
         * Computes the product of two SGal(3) elements following the group operation:
         * (C1, v1, r1, t1) * (C2, v2, r2, t2) = 
	     * (C1*C2, C1*v2 + v1, C1*r2 + v1*t2 + r1, t1 + t2)
         * TODO: (abhi) Need to resolve this. The return type is not 
         * storage agnostic
         */
        template<typename OtherDerived>
        SGal3<Scalar> operator*(const SGal3Base<OtherDerived>& other) const {
            return SGal3<Scalar>(
                rotation() * other.rotation(),
                rotation() * other.translation() + boost() * other.timestamp() + translation(),
                rotation() * other.boost() + boost(),
                timestamp() + other.timestamp()
            );
        }
        /**
         * 
         */
        SGal3<Scalar> inverse() const
        {
            const Sophus::SO3<Scalar> C_inv = rotation().inverse();
            return SGal3<Scalar>(
                C_inv,
                C_inv * (boost() * timestamp() - translation()),
                C_inv * -boost(),
                -timestamp()
            );
        }
    };
    //
    template<typename Scalar, int Options>
    class SGal3 : public SGal3Base<SGal3<Scalar, Options>>
    {
        public:
        //
        using Base = SGal3Base<SGal3<Scalar, Options>>;
        using RotationType = typename Base::RotationType;
        using TranslationType = typename Base::TranslationType;
        using BoostType = typename Base::BoostType;
        using TimestampType = typename Base::TimestampType;
        //
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        //
        static constexpr uint8_t num_parameters = 11;
        //
        static constexpr uint8_t DoF = Base::DoF;
        //
        using TangentVector = Eigen::Matrix<Scalar, DoF, 1>;
        /**
         * 
         */
        // TODO: (abhi): its odd the other constructor doesn't do this
        SGal3() : 
            rotation_(Sophus::SO3<Scalar>()),
            translation_(TranslationType::Zero()),
            boost_(BoostType::Zero()),
            timestamp_(TimestampType::Zero()) {
            static_assert(std::is_standard_layout<SGal3>::value,
                            "Assume standard layout for the use of offsetof check below.");
            static_assert(
                offsetof(SGal3, rotation_) + sizeof(Scalar) * SO3<Scalar>::num_parameters ==
                    offsetof(SGal3, translation_),
                "This class assumes packed storage, in particular"
                "when using [this->data(), this-data() + "
                "num_parameters] to access the raw data in a contiguous fashion.");
            static_assert(
                offsetof(SGal3, translation_) + sizeof(Scalar) * 3 ==
                    offsetof(SGal3, boost_),
                "This class assumes packed storage, in particular"
                "when using [this->data(), this-data() + "
                "num_parameters] to access the raw data in a contiguous fashion.");
            static_assert(
                offsetof(SGal3, boost_) + sizeof(Scalar) * 3 ==
                    offsetof(SGal3, timestamp_),
                "This class assumes packed storage, in particular"
                "when using [this->data(), this-data() + "
                "num_parameters] to access the raw data in a contiguous fashion.");
        }
        /**
         * 
         */
        SGal3(const Eigen::Quaternion<Scalar>& q_,
            const Eigen::Matrix<Scalar, 3, 1> p_,
            const Eigen::Matrix<Scalar, 3, 1> v_,
            const Eigen::Matrix<Scalar, 1, 1> t_) : 
            rotation_(q_),
            translation_(p_),
            boost_(v_),
            timestamp_(t_){};
        /**
         * @brief
         */
        SGal3(const Sophus::SO3<Scalar>& C_,
            const Eigen::Matrix<Scalar, 3, 1> p_,
            const Eigen::Matrix<Scalar, 3, 1> v_,
            const Eigen::Matrix<Scalar, 1, 1> t_) : rotation_(C_),
            translation_(p_),
            boost_(v_),
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
            boost_.setZero();
            timestamp_ = Vector1<Scalar>(0.);
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
         * @brief non-mutable accessor of boost vector
         */
        BoostType const& boost() const {
            return boost_;
        }
        /**
         * @brief mutable accessor of boost vector
         */
        BoostType& boost(){
            return boost_;
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
         * @brief Compute the Galilean E matrix (also called Q in some references)
         * This matrix is used in the exponential and logarithmic maps
         * 
         * @param omega The angular velocity vector (rotation axis * angle)
         * @return The 3x3 Galilean E/Q matrix
         */
        static Eigen::Matrix<Scalar, 3, 3> GalileanQ(
            const Eigen::Matrix<Scalar, 3, 1>& omega)
        {
            Scalar theta = omega.norm();
            Eigen::Matrix<Scalar, 3, 3> W = Sophus::SO3<Scalar>::hat(omega);

            if(theta < 1e-8)
            {
                Eigen::Matrix<Scalar, 3, 3> W_sq = W * W;
                return (Scalar(0.5) * Eigen::Matrix<Scalar, 3, 3>::Identity() + 
                        Scalar(1.0/6.0) * W +
                        Scalar(1.0/24.0) * W_sq);
            }
            else
            {
                Eigen::Matrix<Scalar, 3, 3> W_sq = W * W;
                return (
                    Scalar(0.5) * Eigen::Matrix<Scalar, 3, 3>::Identity()
                    + (theta - sin(theta)) / pow(theta, 3) * W
                    + (theta * theta + Scalar(2.0) * cos(theta) - Scalar(2.0)) / 
                        (Scalar(2.0) * pow(theta, 4)) * W_sq);
            }
        }
        /**
         * @brief Closed-form exponential map for SGal(3)
         * Maps from the Lie algebra sgal(3) (tangent space) to the Lie group SGal(3)
         * 
         * Following Jon Kelly's formulation in "All About the Galilean Group SGal(3)"
         * 
         * @param g_ Tangent vector in sgal(3): [rho, nu, theta, iota]
         * @return SGal(3) group element
         */
        static SGal3 exp(const TangentVector& g_)
        {
            Eigen::Matrix<Scalar, 3, 3> D = 
                Sophus::SO3<Scalar>::leftJacobian(sgal3::theta<Scalar>(g_));
            Eigen::Matrix<Scalar, 3, 3> Q = 
                SGal3::GalileanQ(sgal3::theta<Scalar>(g_));
            //
            // Exponential map formula:
            // C = exp(theta^)
            // r = D*rho + E*nu*iota
            // v = D*nu
            // t = iota
            return SGal3(
                Sophus::SO3<Scalar>::exp(sgal3::theta<Scalar>(g_)),
                D * sgal3::rho<Scalar>(g_) + Q * sgal3::nu<Scalar>(g_) * sgal3::iota<Scalar>(g_),
                D * sgal3::nu<Scalar>(g_),
                sgal3::iota<Scalar>(g_));
        }
        /**
         * @brief Closed-form logarithm map for the SGal(3)
         * Maps from the Lie group SGal(3) to its Lie algebra sgal(3) (tangent space)
         * 
         * Consistent with Jon Kelly's exponential formulation in 
         * "All About the Galilean Group SGal(3)"
         * 
         * @param G SGal(3) group element
         * @return Tangent vector in sgal(3): [rho, nu, theta, iota]
         */
        TangentVector log() const
        {
            TangentVector g;
            ///
            // Extract rotation logarithm
            const Eigen::Vector<Scalar, 3> theta = this->rotation().log();
            //
            // // Compute inverses of Jacobian and Galilean matrix
            const Eigen::Matrix<Scalar, 3, 3> D_inv = Sophus::SO3<Scalar>::leftJacobianInverse(theta);
            const Eigen::Matrix<Scalar, 3, 3> Q = GalileanQ(theta);
            //
            // Logarithmic map formula:
            // nu = D_inv * v
            // rho = D_inv * (r - E*nu*t)
            // theta = log(C)^v
            // iota = t
            const Eigen::Vector<Scalar, 3> nu = D_inv * this->boost();
            g.template segment<3>(0) = D_inv * (this->translation() - Q * nu * this->timestamp());
            g.template segment<3>(3) = nu;
            g.template segment<3>(6) = theta;
            g.template segment<1>(9) = this->timestamp();
            //
            return g;
        }
        //
        protected:
        //
        RotationType rotation_;
        TranslationType translation_;
        BoostType boost_;
        TimestampType timestamp_;
    };
    //
} // namespace Sophus

namespace Eigen
{
    template<typename Scalar, int Options>
    class Map<Sophus::SGal3<Scalar>, Options> : 
        public Sophus::SGal3Base<Map<Sophus::SGal3<Scalar>, Options>>
    {
        using Base = Sophus::SGal3Base<Map<Sophus::SGal3<Scalar>, Options>>;
        using RotationType = typename Base::RotationType;
        using TranslationType = typename Base::TranslationType;
        using BoostType = typename Base::BoostType;
        using TimestampType = typename Base::TimestampType;
        //
        public:
        //
        Map(Scalar* data) :
            rotation_(data),
            // TODO: get num_parameters from RotationType?
            translation_(data + Sophus::SO3<Scalar>::num_parameters),
            boost_(data + Sophus::SO3<Scalar>::num_parameters + 3),
            timestamp_(data + Sophus::SO3<Scalar>::num_parameters + 6)
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
         * @brief non-mutable accessor of boost vector
         */
        BoostType const& boost() const {
            return boost_;
        }
        /**
         * @brief mutable accessor of boost vector
         */
        BoostType& boost(){
            return boost_;
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
        BoostType boost_;
        TimestampType timestamp_;
    };
    /**
     * 
     */
    template<typename Scalar, int Options>
    class Map<Sophus::SGal3<Scalar> const, Options> : 
        public Sophus::SGal3Base<Map<Sophus::SGal3<Scalar> const, Options>>
    {
        public:
        //
        using Base = typename Sophus::SGal3Base<Map<Sophus::SGal3<Scalar> const, Options>>;
        using RotationType = typename Base::RotationType;
        using TranslationType = typename Base::TranslationType;
        using BoostType = typename Base::BoostType;
        using TimestampType = typename Base::TimestampType;
        //
        Map(Scalar const* data) :
            rotation_(data),
            // TODO: get num_parameters from RotationType?
            translation_(data + Sophus::SO3<Scalar>::num_parameters),
            boost_(data + Sophus::SO3<Scalar>::num_parameters + 3),
            timestamp_(data + Sophus::SO3<Scalar>::num_parameters + 6)
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
         * @brief non-mutable accessor of boost vector
         */
        BoostType const& boost() const {
            return boost_;
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
        BoostType boost_;
        TimestampType timestamp_;
    };
};
