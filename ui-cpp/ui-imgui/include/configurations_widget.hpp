#pragma once
#ifndef SPIRIT_IMGUI_CONFIGURATIONS_WIDGET_HPP
#define SPIRIT_IMGUI_CONFIGURATIONS_WIDGET_HPP

#include <rendering_layer.hpp>
#include <widget_base.hpp>

#include <memory>

struct State;

namespace ui
{

struct ConfigurationsWidget : public WidgetBase
{
    ConfigurationsWidget(
        bool & show, std::shared_ptr<State> state, UiSharedState & ui_shared_state, RenderingLayer & rendering_layer );

    void reset_settings();
    void show_content() override;
    void update_data();

    void set_plus_z();
    void set_minus_z();
    void set_random();
    void set_spiral();
    void set_skyrmion();
    void set_hopfion();

    std::shared_ptr<State> state;
    RenderingLayer & rendering_layer;
    UiSharedState & ui_shared_state;

    scalar sk_radius = 15;
    scalar sk_speed  = 1;
    scalar sk_phase  = 0;
    bool sk_up_down  = false;
    bool sk_achiral  = false;
    bool sk_rl       = false;

    scalar hopfion_radius = 10;
    int hopfion_order     = 1;

    scalar spiral_angle    = 0;
    scalar spiral_axis[3]  = { 0, 0, 1 };
    scalar spiral_qmag     = 1;
    scalar spiral_qvec[3]  = { 1, 0, 0 };
    bool spiral_q2         = false;
    scalar spiral_qmag2    = 1;
    scalar spiral_qvec2[3] = { 1, 0, 0 };

    int transition_idx_1 = 1;
    int transition_idx_2 = 1;
};

} // namespace ui

#endif