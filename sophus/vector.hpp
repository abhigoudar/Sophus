#pragma once
#include <sophus/types.hpp>

namespace se_opt
{
    // TODO: Move to traits
    // template<typename Scalar_, size_t DoF_>
    // class Vector : public Sophus::Vector<Scalar_, DoF_>
    // {
    //     public:
    //     //
    //     using Base = typename Sophus::Vector<Scalar_, DoF_>;
    //     using Scalar = Scalar_;
    //     static constexpr size_t DoF = DoF_;
    //     static constexpr size_t num_parameters = DoF_;
    //     //
    //     void weightedIncrement(const Scalar weight,const Vector<Scalar, DoF_>& val)
    //     {
    //        *this += weight * val;
    //     }
    // };
    // //
    // using Vector1d = Vector<double, 1>;
    // using Vector2d = Vector<double, 2>;
    // using Vector3d = Vector<double, 3>;
}