#include <fonts.hpp>
#include <styles.hpp>
#include <widgets.hpp>

#include <Spirit/Simulation.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>

#include <imgui/imgui.h>

#include <imgui-gizmo3d/imGuIZMOquat.h>

#include <fmt/format.h>

#include <map>

namespace widgets
{

void show_plots( bool & show )
{
    auto & style = ImGui::GetStyle();

    static bool plot_image_energies        = true;
    static bool plot_interpolated_energies = true;

    static bool animate = true;
    // static float values[90]    = {};
    static std::vector<float> energies( 90, 0 );
    static int values_offset   = 0;
    static double refresh_time = 0.0;
    if( !animate || refresh_time == 0.0 )
        refresh_time = ImGui::GetTime();
    while( refresh_time < ImGui::GetTime() ) // Create dummy data at fixed 60 Hz rate for the demo
    {
        static float phase      = 0.0f;
        energies[values_offset] = cosf( phase );
        values_offset           = ( values_offset + 1 ) % energies.size();
        phase += 0.10f * values_offset;
        refresh_time += 1.0f / 60.0f;
    }

    if( !show )
        return;

    ImGui::Begin( "Plots", &show );
    {
        ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
        if( ImGui::BeginTabBar( "plots_tab_bar", tab_bar_flags ) )
        {
            if( ImGui::BeginTabItem( "Energy" ) )
            {
                std::string overlay = fmt::format( "{:.3e}", energies[energies.size() - 1] );

                // ImGui::Text( "E" );
                // ImGui::SameLine();
                ImGui::PlotLines(
                    "", energies.data(), energies.size(), values_offset, overlay.c_str(), -1.0f, 1.0f,
                    ImVec2(
                        ImGui::GetWindowContentRegionMax().x - 2 * style.FramePadding.x,
                        ImGui::GetWindowContentRegionMax().y - 110.f ) );

                ImGui::Checkbox( "Image energies", &plot_image_energies );
                ImGui::Checkbox( "Interpolated energies", &plot_interpolated_energies );
                ImGui::SameLine();
                static bool inputs_step = true;
                const ImU32 u32_one     = (ImU32)1;
                static ImU32 u32_v      = (ImU32)10;
                ImGui::PushItemWidth( 100 );
                ImGui::InputScalar(
                    "##energies", ImGuiDataType_U32, &u32_v, inputs_step ? &u32_one : NULL, NULL, "%u" );
                ImGui::PopItemWidth();

                ImGui::EndTabItem();
            }
            if( ImGui::BeginTabItem( "Convergence" ) )
            {
                std::string overlay = fmt::format( "{:.3e}", energies[energies.size() - 1] );

                // ImGui::Text( "F" );
                // ImGui::SameLine();
                ImGui::PlotLines(
                    "", energies.data(), energies.size(), values_offset, overlay.c_str(), -1.0f, 1.0f,
                    ImVec2(
                        ImGui::GetWindowContentRegionMax().x - 2 * style.FramePadding.x,
                        ImGui::GetWindowContentRegionMax().y - 90.f ) );

                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}

void show_parameters( bool & show, GUI_Mode & selected_mode )
{
    if( !show )
        return;

    ImGui::Begin( "Parameters", &show );

    if( selected_mode == GUI_Mode::Minimizer )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }
    else if( selected_mode == GUI_Mode::MC )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }
    else if( selected_mode == GUI_Mode::LLG )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }
    else if( selected_mode == GUI_Mode::GNEB )
    {
    }
    else if( selected_mode == GUI_Mode::MMF )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }
    else if( selected_mode == GUI_Mode::EMA )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }

    ImGui::End();
}

void show_visualisation_settings( bool & show, VFRendering::View & vfr_view, glm::vec4 & background_colour )
{
    if( !show )
        return;

    ImGui::Begin( "Visualisation settings", &show );

    ImGui::Text( "Background color" );
    if( ImGui::ColorEdit3( "##bgcolour", (float *)&background_colour ) )
    {
        vfr_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>( background_colour );
    }

    ImGui::Separator();

    static vgm::Vec3 dir( 0, 0, -1 );
    ImGui::Text( "Light direction" );
    ImGui::Columns( 2, "lightdircolumns", false ); // 3-ways, no border
    if( ImGui::gizmo3D( "##dir", dir ) )
    {
        vfr_view.setOption<VFRendering::View::Option::LIGHT_POSITION>(
            { -1000 * dir.x, -1000 * dir.y, -1000 * dir.z } );
    }
    ImGui::NextColumn();
    ImGui::Text( fmt::format( "{:>6.3f}", dir.x ).c_str() );
    ImGui::Text( fmt::format( "{:>6.3f}", dir.y ).c_str() );
    ImGui::Text( fmt::format( "{:>6.3f}", dir.z ).c_str() );
    ImGui::Columns( 1 );
    ImGui::End();
}

void show_overlay_system( bool & show )
{
    if( !show )
        return;

    static float energy = 0;
    static float m_x    = 0;
    static float m_y    = 0;
    static float m_z    = 0;

    static int noi           = 1;
    static int nos           = 1;
    static int n_basis_atoms = 1;
    static int n_a           = 1;
    static int n_b           = 1;
    static int n_c           = 1;

    const float DISTANCE = 50.0f;
    static int corner    = 0;

    ImGuiIO & io = ImGui::GetIO();

    if( corner != -1 )
    {
        ImVec2 window_pos = ImVec2(
            ( corner & 1 ) ? io.DisplaySize.x - DISTANCE : DISTANCE,
            ( corner & 2 ) ? io.DisplaySize.y - DISTANCE : DISTANCE );
        ImVec2 window_pos_pivot = ImVec2( ( corner & 1 ) ? 1.0f : 0.0f, ( corner & 2 ) ? 1.0f : 0.0f );
        ImGui::SetNextWindowPos( window_pos, ImGuiCond_Always, window_pos_pivot );
    }
    ImGui::SetNextWindowBgAlpha( 0.35f ); // Transparent background
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                    | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                    | ImGuiWindowFlags_NoNav;
    if( corner != -1 )
        window_flags |= ImGuiWindowFlags_NoMove;
    if( ImGui::Begin( "System information overlay", &show, window_flags ) )
    {
        ImGui::Text( fmt::format( "FPS: {:d}", int( io.Framerate ) ).c_str() );

        ImGui::Separator();

        ImGui::Text( fmt::format( "E      = {:.10f}", energy ).c_str() );
        ImGui::Text( fmt::format( "E dens = {:.10f}", energy / nos ).c_str() );

        ImGui::Separator();

        ImGui::Text( fmt::format( "M_x = {:.8f}", m_x ).c_str() );
        ImGui::Text( fmt::format( "M_y = {:.8f}", m_y ).c_str() );
        ImGui::Text( fmt::format( "M_z = {:.8f}", m_z ).c_str() );

        ImGui::Separator();

        ImGui::Text( fmt::format( "NOI: {}", noi ).c_str() );
        ImGui::Text( fmt::format( "NOS: {}", nos ).c_str() );
        ImGui::Text( fmt::format( "N basis atoms: {}", n_basis_atoms ).c_str() );
        ImGui::Text( fmt::format( "Cells: {}x{}x{}", n_a, n_b, n_c ).c_str() );

        ImGui::Separator();

        ImGui::Text( "Simple overlay\n"
                     "in the corner of the screen.\n"
                     "(right-click to change position)" );

        ImGui::Separator();

        if( ImGui::IsMousePosValid() )
            ImGui::Text( "Mouse Position: (%.1f,%.1f)", io.MousePos.x, io.MousePos.y );
        else
            ImGui::Text( "Mouse Position: <invalid>" );

        if( ImGui::BeginPopupContextWindow() )
        {
            if( ImGui::MenuItem( "Custom", NULL, corner == -1 ) )
                corner = -1;
            if( ImGui::MenuItem( "Top-left", NULL, corner == 0 ) )
                corner = 0;
            if( ImGui::MenuItem( "Top-right", NULL, corner == 1 ) )
                corner = 1;
            if( ImGui::MenuItem( "Bottom-left", NULL, corner == 2 ) )
                corner = 2;
            if( ImGui::MenuItem( "Bottom-right", NULL, corner == 3 ) )
                corner = 3;
            if( show && ImGui::MenuItem( "Close" ) )
                show = false;
            ImGui::EndPopup();
        }
    }
    ImGui::End();
}

void show_overlay_calculation(
    bool & show, GUI_Mode & selected_mode, int & selected_solver_min, int & selected_solver_llg )
{
    if( !show )
        return;

    static auto solvers_llg
        = std::map<int, std::pair<std::string, std::string>>{ { Solver_SIB, { "SIB", "Semi-implicit method B" } },
                                                              { Solver_Depondt, { "Depondt", "Depondt" } },
                                                              { Solver_Heun, { "Heun", "Heun" } },
                                                              { Solver_RungeKutta4,
                                                                { "RK4", "4th order Runge-Kutta" } } };

    static auto solvers_min = std::map<int, std::pair<std::string, std::string>>{
        { Solver_VP, { "VP", "Velocity Projection" } },
        { Solver_VP_OSO, { "VP (OSO)", "Velocity Projection (OSO)" } },
        { Solver_LBFGS_OSO, { "LBFGS (OSO)", "LBFGS (OSO)" } },
        { Solver_LBFGS_Atlas, { "LBFGS (Atlas)", "LBFGS (Atlas)" } },
        { Solver_SIB, { "SIB", "Semi-implicit method B" } },
        { Solver_Depondt, { "Depondt", "Depondt" } },
        { Solver_Heun, { "Heun", "Heun" } },
        { Solver_RungeKutta4, { "RK4", "4th order Runge-Kutta" } }
    };

    static float simulated_time = 0;
    static float wall_time      = 0;
    static int iteration        = 0;
    static float ips            = 0;

    int hours        = wall_time / ( 60 * 60 * 1000 );
    int minutes      = ( wall_time - 60 * 60 * 1000 * hours ) / ( 60 * 1000 );
    int seconds      = ( wall_time - 60 * 60 * 1000 * hours - 60 * 1000 * minutes ) / 1000;
    int milliseconds = wall_time - 60 * 60 * 1000 * hours - 60 * 1000 * minutes - 1000 * seconds;

    static float force_max = 0;

    // ips = Simulation_Get_IterationsPerSecond( state.get() );

    const float DISTANCE = 50.0f;
    static int corner    = 1;

    ImGuiIO & io = ImGui::GetIO();
    if( corner != -1 )
    {
        ImVec2 window_pos = ImVec2(
            ( corner & 1 ) ? io.DisplaySize.x - DISTANCE : DISTANCE,
            ( corner & 2 ) ? io.DisplaySize.y - DISTANCE : DISTANCE );
        ImVec2 window_pos_pivot = ImVec2( ( corner & 1 ) ? 1.0f : 0.0f, ( corner & 2 ) ? 1.0f : 0.0f );
        ImGui::SetNextWindowPos( window_pos, ImGuiCond_Always, window_pos_pivot );
    }
    ImGui::SetNextWindowBgAlpha( 0.35f ); // Transparent background
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                    | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                    | ImGuiWindowFlags_NoNav;
    if( corner != -1 )
        window_flags |= ImGuiWindowFlags_NoMove;
    if( ImGui::Begin( "Calculation information overlay", &show, window_flags ) )
    {
        if( selected_mode == GUI_Mode::Minimizer || selected_mode == GUI_Mode::GNEB || selected_mode == GUI_Mode::MMF )
        {
            if( ImGui::Button( fmt::format( "Solver: {}", solvers_min[selected_solver_min].first ).c_str() ) )
                ImGui::OpenPopup( "solver_popup_min" );
            if( ImGui::BeginPopup( "solver_popup_min" ) )
            {
                for( auto solver : solvers_min )
                    if( ImGui::Selectable( solver.second.first.c_str() ) )
                        selected_solver_min = solver.first;
                ImGui::EndPopup();
            }
            ImGui::Separator();
        }
        else if( selected_mode == GUI_Mode::LLG )
        {
            if( ImGui::Button( fmt::format( "Solver: {}", solvers_llg[selected_solver_llg].first ).c_str() ) )
                ImGui::OpenPopup( "solver_popup_llg" );
            if( ImGui::BeginPopup( "solver_popup_llg" ) )
            {
                for( auto solver : solvers_llg )
                    if( ImGui::Selectable( solver.second.first.c_str() ) )
                        selected_solver_llg = solver.first;
                ImGui::EndPopup();
            }
            ImGui::Separator();
            ImGui::Text( fmt::format( "t = {} ps", simulated_time ).c_str() );
        }

        ImGui::Text( fmt::format( "{:0>2d}:{:0>2d}:{:0>2d}.{:0>3d}", hours, minutes, seconds, milliseconds ).c_str() );
        ImGui::Text( fmt::format( "Iteration: {}", iteration ).c_str() );
        ImGui::Text( fmt::format( "IPS: {:.2f}", ips ).c_str() );

        ImGui::Separator();

        ImGui::Text( fmt::format( "F_max = {:.5e}", force_max ).c_str() );
        if( selected_mode == GUI_Mode::GNEB )
        {
            ImGui::Text( fmt::format( "F_current = {:.5e}", simulated_time ).c_str() );
        }

        if( ImGui::BeginPopupContextWindow() )
        {
            if( ImGui::MenuItem( "Custom", NULL, corner == -1 ) )
                corner = -1;
            if( ImGui::MenuItem( "Top-left", NULL, corner == 0 ) )
                corner = 0;
            if( ImGui::MenuItem( "Top-right", NULL, corner == 1 ) )
                corner = 1;
            if( ImGui::MenuItem( "Bottom-left", NULL, corner == 2 ) )
                corner = 2;
            if( ImGui::MenuItem( "Bottom-right", NULL, corner == 3 ) )
                corner = 3;
            if( show && ImGui::MenuItem( "Close" ) )
                show = false;
            ImGui::EndPopup();
        }
    }
    ImGui::End();
}

void show_keybindings( bool & show )
{
    if( !show )
        return;

    ImGui::Begin( "Keybindings", &show );

    ImGui::Text( "UI controls" );
    ImGui::BulletText( "F1: Show this" );
    ImGui::BulletText( "F2: Toggle settings" );
    ImGui::BulletText( "F3: Toggle plots" );
    ImGui::BulletText( "F4: Toggle debug" );
    ImGui::BulletText( "F5: Toggle \"Dragging\" mode" );
    ImGui::BulletText( "F6: Toggle \"Defects\" mode" );
    ImGui::BulletText( "F7: Toggle \"Pinning\" mode" );
    ImGui::Text( "" );
    ImGui::BulletText( "F10  and Ctrl+F:      Toggle large visualisation" );
    ImGui::BulletText( "F11 and Ctrl+Shift+F: Toggle fullscreen window" );
    ImGui::BulletText( "F12 and Home:         Screenshot of visualisation region" );
    ImGui::BulletText( "Ctrl+Shift+V:         Toggle OpenGL visualisation" );
    ImGui::BulletText( "i:                    Toggle large visualisation" );
    ImGui::BulletText( "Escape:               Try to return focus to main UI (does not always work)" );
    ImGui::Text( "" );
    ImGui::Text( "Camera controls" );
    ImGui::BulletText( "Left mouse:   Rotate the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "Right mouse:  Move the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "Scroll mouse: Zoom in on focus point (<b>shift</b> to go slow)" );
    ImGui::BulletText( "WASD:         Rotate the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "TFGH:         Move the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "X,Y,Z:        Set the camera in X, Y or Z direction (<b>shift</b> to invert)" );
    ImGui::Text( "" );
    ImGui::Text( "Control Simulations" );
    ImGui::BulletText( "Space:        Start/stop calculation" );
    ImGui::BulletText( "Ctrl+M:       Cycle method" );
    ImGui::BulletText( "Ctrl+S:       Cycle solver" );
    ImGui::Text( "" );
    ImGui::Text( "Manipulate the current images" );
    ImGui::BulletText( "Ctrl+R:       Random configuration" );
    ImGui::BulletText( "Ctrl+N:       Add tempered noise" );
    ImGui::BulletText( "Enter:        Insert last used configuration" );
    ImGui::Text( "" );
    ImGui::Text( "Visualisation" );
    ImGui::BulletText( "+/-:          Use more/fewer data points of the vector field" );
    ImGui::BulletText( "1:            Regular Visualisation Mode" );
    ImGui::BulletText( "2:            Isosurface Visualisation Mode" );
    ImGui::BulletText( "3-5:          Slab (X,Y,Z) Visualisation Mode" );
    ImGui::BulletText( "/:            Cycle Visualisation Mode" );
    ImGui::BulletText( ", and .:      Move Slab (<b>shift</b> to go faster)" );
    ImGui::Text( "" );
    ImGui::Text( "Manipulate the chain of images" );
    ImGui::BulletText( "Arrows:          Switch between images and chains" );
    ImGui::BulletText( "Ctrl+X:          Cut   image" );
    ImGui::BulletText( "Ctrl+C:          Copy  image" );
    ImGui::BulletText( "Ctrl+V:          Paste image at current index" );
    ImGui::BulletText( "Ctrl+Left/Right: Insert left/right of current index<" );
    ImGui::BulletText( "Del:             Delete image" );
    ImGui::Text( "" );
    ImGui::TextWrapped( "Note that some of the keybindings may only work correctly on US keyboard layout.\n"
                        "\n"
                        "For more information see the documentation at spirit-docs.readthedocs.io" );
    ImGui::Text( "" );

    if( ImGui::Button( "Close" ) )
        show = false;
    ImGui::End();
}

void show_about( bool & show_about )
{
    if( !show_about )
        return;

    ImGui::Begin( fmt::format( "About Spirit {}", Spirit_Version() ).c_str() );

    ImGui::TextWrapped( "The <b>Spirit</b> GUI application incorporates intuitive visualisation,"
                        "powerful <b>Spin Dynamics</b> and <b>Nudged Elastic Band</b> tools"
                        "into a cross-platform user interface." );

    ImGui::Text( "" );
    ImGui::Separator();
    ImGui::Text( "" );

    ImGui::Text( "Main developers:" );
    ImGui::BulletText(
        "Moritz Sallermann (<a href=\"mailto:m.sallermann@fz-juelich.de\">m.sallermann@fz-juelich.de</a>)" );
    ImGui::BulletText( "Gideon Mueller (<a href=\"mailto:g.mueller@fz-juelich.de\">g.mueller@fz-juelich.de</a>)" );
    ImGui::TextWrapped(
        "at the Institute for Advanced Simulation 1 of the Forschungszentrum Juelich.\n"
        "For more information about us, visit <a href=\"http://juspin.de\">juSpin.de</a>"
        " or see the <a href=\"http://www.fz-juelich.de/pgi/pgi-1/DE/Home/home_node.html\">IAS-1 Website</a>." );

    ImGui::Text( "" );

    ImGui::TextWrapped( "The sources are hosted at <a href=\"https://spirit-code.github.io\">spirit-code.github.io</a>"
                        " and the documentation can be found at <a "
                        "href=\"https://spirit-docs.readthedocs.io\">spirit-docs.readthedocs.io</a>." );

    ImGui::Text( "" );
    ImGui::Separator();
    ImGui::Text( "" );

    ImGui::Text( fmt::format( "Full library version {}", Spirit_Version_Full() ).c_str() );
    ImGui::Text( fmt::format( "Built with {}", Spirit_Compiler_Full() ).c_str() );

    ImGui::Text( "" );

    ImGui::Text( fmt::format( "Floating point precision = {}", Spirit_Scalar_Type() ).c_str() );

    ImGui::Text( "" );

    ImGui::Columns( 2, "aboutinfocolumns", false );

    ImGui::Text( "Parallelisation:" );
    ImGui::Text( fmt::format( "   - OpenMP  = {}", Spirit_OpenMP() ).c_str() );
    ImGui::Text( fmt::format( "   - Cuda    = {}", Spirit_Cuda() ).c_str() );
    ImGui::Text( fmt::format( "   - Threads = {}", Spirit_Threads() ).c_str() );
    ImGui::NextColumn();
    ImGui::Text( "Other:" );
    ImGui::Text( fmt::format( "   - Defects = {}", Spirit_Defects() ).c_str() );
    ImGui::Text( fmt::format( "   - Pinning = {}", Spirit_Pinning() ).c_str() );
    ImGui::Text( fmt::format( "   - FFTW    = {}", Spirit_FFTW() ).c_str() );
    ImGui::Columns( 1 );

    ImGui::Text( "" );

    if( ImGui::Button( "Close" ) )
        show_about = false;
    ImGui::End();
}

// Helper to display a little (?) mark which shows a tooltip when hovered.
// In your own code you may want to display an actual icon if you are using a merged icon fonts (see docs/FONTS.txt)
void help_marker( const char * description )
{
    ImGui::TextDisabled( "(?)" );
    if( ImGui::IsItemHovered() )
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos( ImGui::GetFontSize() * 35.0f );
        ImGui::TextUnformatted( description );
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

} // namespace widgets