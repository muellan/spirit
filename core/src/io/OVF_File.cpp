#include <io/OVF_File.hpp>

#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <engine/Vectormath.hpp>

using namespace Utility;

namespace IO
{
    OVF_File::OVF_File( std::string filename, VF_FileFormat format, const std::string comment ) : 
        filename(filename), format(format), comment(comment), myfile(filename, format) 
    {
        this->output_to_file = "";
        this->output_to_file.reserve( int( 0x08000000 ) );  // reserve 128[MByte]
        this->empty_line = "#\n";
        this->sender = Log_Sender::IO;
        
        if ( this->format == VF_FileFormat::OVF_BIN8 ) 
            this->datatype_out = "Binary 8";
        else if ( this->format == VF_FileFormat::OVF_BIN4 )
            this->datatype_out = "Binary 4";
        else if( this->format == VF_FileFormat::OVF_TEXT )
            this->datatype_out = "Text";
        
        this->version = "";
        this->title = "";
        this->meshunit = "";
        this->meshtype = "";
        this->valueunits = "";
        this->datatype_in = "";
        this->max = Vector3(0,0,0);
        this->min = Vector3(0,0,0);
        this->base = Vector3(0,0,0);
        this->stepsize = Vector3(0,0,0);
    }

    void OVF_File::Write_Top_Header( const int n_segments )
    {
        this->n_segments = n_segments; 
        
        this->output_to_file += fmt::format( "# OOMMF OVF 2.0\n" );
        this->output_to_file += fmt::format( this->empty_line );
        
        this->output_to_file += fmt::format( "# Segment count: {}\n", this->n_segments );
        this->output_to_file += fmt::format( this->empty_line );
        
        Dump_to_File( this->output_to_file, this->filename );  // Dump to file
        this->output_to_file = "";  // reset output string buffer
    }
    
    void OVF_File::Write_Segment( const vectorfield& vf, const Data::Geometry& geometry)
    {
        this->output_to_file += fmt::format( "# Begin: Segment\n" );
        this->output_to_file += fmt::format( "# Begin: Header\n" );
        this->output_to_file += fmt::format( this->empty_line );
        
        this->output_to_file += fmt::format( "# Title: SPIRIT Version {}\n", Utility::version_full );
        this->output_to_file += fmt::format( this->empty_line );
        
        this->output_to_file += fmt::format( "# Desc: {}\n", this->comment );
        this->output_to_file += fmt::format( this->empty_line );
        
        // The value dimension is always 3 in this implementation since we are writting Vector3-data
        this->output_to_file += fmt::format( "# valuedim: {} ##Value dimension\n", 3 );
        this->output_to_file += fmt::format( "# valueunits: None None None\n" );
        this->output_to_file += fmt::format( "# valuelabels: spin_x_component spin_y_component "
                                       "spin_z_component \n" );
        this->output_to_file += fmt::format( this->empty_line );
        
        this->output_to_file += fmt::format( "## Fundamental mesh measurement unit. "
                                       "Treated as a label:\n" );
        this->output_to_file += fmt::format( "# meshunit: nm\n" );                  //// TODO: treat that
        this->output_to_file += fmt::format( this->empty_line );
        
        this->output_to_file += fmt::format( "# xmin: {}\n", geometry.bounds_min[0] );
        this->output_to_file += fmt::format( "# ymin: {}\n", geometry.bounds_min[1] );
        this->output_to_file += fmt::format( "# zmin: {}\n", geometry.bounds_min[2] );
        this->output_to_file += fmt::format( "# xmax: {}\n", geometry.bounds_max[0] );
        this->output_to_file += fmt::format( "# ymax: {}\n", geometry.bounds_max[1] );
        this->output_to_file += fmt::format( "# zmax: {}\n", geometry.bounds_max[2] );
        this->output_to_file += fmt::format( this->empty_line );
        
        // TODO: Spirit does not support irregular geometry yet. We are emmiting rectangular mesh
        this->output_to_file += fmt::format( "# meshtype: rectangular\n" );
        
        // TODO: maybe this is not true for every system
        this->output_to_file += fmt::format( "# xbase: {}\n", 0 );
        this->output_to_file += fmt::format( "# ybase: {}\n", 0 );
        this->output_to_file += fmt::format( "# zbase: {}\n", 0 );
        
        this->output_to_file += fmt::format( "# xstepsize: {}\n", 
                                       geometry.lattice_constant * geometry.bravais_vectors[0][0] );
        this->output_to_file += fmt::format( "# ystepsize: {}\n", 
                                       geometry.lattice_constant * geometry.bravais_vectors[1][1] );
        this->output_to_file += fmt::format( "# zstepsize: {}\n", 
                                       geometry.lattice_constant * geometry.bravais_vectors[2][2] );
        
        this->output_to_file += fmt::format( "# xnodes: {}\n", geometry.n_cells[0] );
        this->output_to_file += fmt::format( "# ynodes: {}\n", geometry.n_cells[1] );
        this->output_to_file += fmt::format( "# znodes: {}\n", geometry.n_cells[2] );
        this->output_to_file += fmt::format( this->empty_line );
        
        this->output_to_file += fmt::format( "# End: Header\n" );
        this->output_to_file += fmt::format( this->empty_line );
        
        // Data
        this->output_to_file += fmt::format( "# Begin: Data {}\n", this->datatype_out );
        
        if ( format == VF_FileFormat::OVF_BIN8 || format == VF_FileFormat::OVF_BIN4 )
            Write_Data_bin( vf );
        else if ( format == VF_FileFormat::OVF_TEXT )
            Write_Data_txt( vf );
        
        this->output_to_file += fmt::format( "# End: Data {}\n", this->datatype_out );
        
        this->output_to_file += fmt::format( "# End: Segment\n" );
        
        Append_String_to_File( this->output_to_file, this->filename );  // Append the #End keywords
        this->output_to_file = "";  // reset output string buffer
    }

    void OVF_File::Write_Data_bin( const vectorfield& vf )
    {
        // float test value
        const float ref_4b = *reinterpret_cast<const float *>( &this->test_hex_4b );
        
        // double test value
        const double ref_8b = *reinterpret_cast<const double *>( &this->test_hex_8b );
        
        if( format == VF_FileFormat::OVF_BIN8 )
        {
            this->output_to_file += std::string( reinterpret_cast<const char *>(&ref_8b),
                sizeof(double) );
            
            // in case that scalar is 4bytes long
            if (sizeof(scalar) == sizeof(float))
            {
                double buffer[3];
                for (unsigned int i=0; i<vf.size(); i++)
                {
                    buffer[0] = static_cast<double>(vf[i][0]);
                    buffer[1] = static_cast<double>(vf[i][1]);
                    buffer[2] = static_cast<double>(vf[i][2]);
                    this->output_to_file += std::string( reinterpret_cast<char *>(buffer), 
                        sizeof(buffer) );
                }
            } 
            else
            {
                for (unsigned int i=0; i<vf.size(); i++)
                    this->output_to_file += 
                        std::string( reinterpret_cast<const char *>(&vf[i]), 3*sizeof(double) );
            }
        }
        else if( format == VF_FileFormat::OVF_BIN4 )
        {
            this->output_to_file += std::string( reinterpret_cast<const char *>(&ref_4b),
                sizeof(float) );
            
            // in case that scalar is 8bytes long
            if (sizeof(scalar) == sizeof(double))
            {
                float buffer[3];
                for (unsigned int i=0; i<vf.size(); i++)
                {
                    buffer[0] = static_cast<float>(vf[i][0]);
                    buffer[1] = static_cast<float>(vf[i][1]);
                    buffer[2] = static_cast<float>(vf[i][2]);
                    this->output_to_file += std::string( reinterpret_cast<char *>(buffer), 
                        sizeof(buffer) );
                }
            } 
            else
            {
                for (unsigned int i=0; i<vf.size(); i++)
                    this->output_to_file += 
                        std::string( reinterpret_cast<const char *>(&vf[i]), 3*sizeof(float) );
            }
        }
    }

    void OVF_File::Write_Data_txt( const vectorfield& vf )
    {
        for (int iatom = 0; iatom < vf.size(); ++iatom)
        {
                this->output_to_file += fmt::format( "{:20.10f} {:20.10f} {:20.10f}\n", 
                                                      vf[iatom][0], vf[iatom][1], vf[iatom][2] );
        }
    }

    
    void OVF_File::Read_N_Segments()
    {
    }

    void OVF_File::Read_Header()
    {
        try
        {
            // first line - OVF version
            myfile.Read_String( this->version, "# OOMMF OVF" );
            if( this->version != "2.0" && this->version != "2" )
            {
                spirit_throw( Utility::Exception_Classifier::Bad_File_Content, 
                              Utility::Log_Level::Error,
                              fmt::format( "OVF {0} is not supported", this->version ) );
            }
            
            // Title
            myfile.Read_String( this->title, "# Title:" );
            
            // mesh units 
            myfile.Read_Single( this->meshunit, "# meshunit:" );
            
            // value's dimensions
            myfile.Require_Single( this->valuedim, "# valuedim:" );
            
            // value's units
            myfile.Read_String( this->valueunits, "# valueunits:" );
            
            // value's labels
            myfile.Read_String( this->valueunits, "# valuelabels:" );
            
            // {x,y,z} x {min,max}
            myfile.Read_Single( this->min.x(), "# xmin:" );
            myfile.Read_Single( this->min.y(), "# ymin:" );
            myfile.Read_Single( this->min.z(), "# zmin:" );
            myfile.Read_Single( this->max.x(), "# xmax:" );
            myfile.Read_Single( this->max.y(), "# ymax:" );
            myfile.Read_Single( this->max.z(), "# zmax:" );
            
            // meshtype
            myfile.Require_Single( this->meshtype, "# meshtype:" );
            
            // TODO: Change the throw to something more meaningfull we don't need want termination
            if( this->meshtype != "rectangular" && this->meshtype != "irregular" )
            {
                spirit_throw(Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    "Mesh type must be either \"rectangular\" or \"irregular\"");
            }
            
            // Emit Header to Log
            auto lvl = Log_Level::Parameter;
            
            Log( lvl, this->sender, fmt::format( "# OVF version             = {}", this->version ) );
            Log( lvl, this->sender, fmt::format( "# OVF title               = {}", this->title ) );
            Log( lvl, this->sender, fmt::format( "# OVF values dimensions   = {}", this->valuedim ) );
            Log( lvl, this->sender, fmt::format( "# OVF meshunit            = {}", this->meshunit ) );
            Log( lvl, this->sender, fmt::format( "# OVF xmin                = {}", this->min.x() ) );
            Log( lvl, this->sender, fmt::format( "# OVF ymin                = {}", this->min.y() ) );
            Log( lvl, this->sender, fmt::format( "# OVF zmin                = {}", this->min.z() ) );
            Log( lvl, this->sender, fmt::format( "# OVF xmax                = {}", this->max.x() ) );
            Log( lvl, this->sender, fmt::format( "# OVF ymax                = {}", this->max.y() ) );
            Log( lvl, this->sender, fmt::format( "# OVF zmax                = {}", this->max.z() ) );
            
            // For different mesh types
            if( this->meshtype == "rectangular" )
            {
                // {x,y,z} x {base,stepsize,nodes} 
                
                myfile.Require_Single( this->base.x(), "# xbase:" );
                myfile.Require_Single( this->base.y(), "# ybase:" );
                myfile.Require_Single( this->base.z(), "# zbase:" );
                
                myfile.Require_Single( this->stepsize.x(), "# xstepsize:" );
                myfile.Require_Single( this->stepsize.y(), "# ystepsize:" );
                myfile.Require_Single( this->stepsize.z(), "# zstepsize:" );
                
                myfile.Require_Single( this->nodes[0], "# xnodes:" );
                myfile.Require_Single( this->nodes[1], "# ynodes:" );
                myfile.Require_Single( this->nodes[2], "# znodes:" );
                
                // Write to Log
                Log( lvl, this->sender, fmt::format( "# OVF meshtype <{}>", this->meshtype ) );
                Log( lvl, this->sender, fmt::format( "# xbase      = {:.8f}", this->base.x() ) );
                Log( lvl, this->sender, fmt::format( "# ybase      = {:.8f}", this->base.y() ) );
                Log( lvl, this->sender, fmt::format( "# zbase      = {:.8f}", this->base.z() ) );
                Log( lvl, this->sender, fmt::format( "# xstepsize  = {:.8f}", this->stepsize.x() ) );
                Log( lvl, this->sender, fmt::format( "# ystepsize  = {:.8f}", this->stepsize.y() ) );
                Log( lvl, this->sender, fmt::format( "# zstepsize  = {:.8f}", this->stepsize.z() ) );
                Log( lvl, this->sender, fmt::format( "# xnodes     = {}", this->nodes[0] ) );
                Log( lvl, this->sender, fmt::format( "# ynodes     = {}", this->nodes[1] ) );
                Log( lvl, this->sender, fmt::format( "# znodes     = {}", this->nodes[2] ) );
            }
            
            // Check mesh type
            if ( this->meshtype == "irregular" )
            {
                // pointcount
                myfile.Require_Single( this->pointcount, "# pointcount:" );
                
                // Write to Log
                Log( lvl, this->sender, fmt::format( "# OVF meshtype <{}>", this->meshtype ) );
                Log( lvl, this->sender, fmt::format( "# OVF point count = {}", this->pointcount ) );
            }
        }
        catch (...) 
        {
            spirit_rethrow( fmt::format("Failed to read OVF file \"{}\".", this->filename) );
        }
    }
    
    void OVF_File::Read_Check_Geometry( const Data::Geometry& geometry)
    {
        try
        {
            // Check that nos is smaller or equal to the nos of the current image
            int nos = this->nodes[0] * this->nodes[1] * this->nodes[2];
            if ( nos > geometry.nos )
                spirit_throw(Utility::Exception_Classifier::Bad_File_Content, 
                    Utility::Log_Level::Error,"NOS of the OVF file is greater than the NOS in the "
                    "current image");
            
            // Check if the geometry of the ovf file is the same with the one of the current image
            if ( this->nodes[0] != geometry.n_cells[0] ||
                 this->nodes[1] != geometry.n_cells[1] ||
                 this->nodes[2] != geometry.n_cells[2] )
            {
                Log(Log_Level::Warning, this->sender, fmt::format("The geometry of the OVF file "
                    "does not much the geometry of the current image") );
            }
        }
        catch (...) 
        {
            spirit_rethrow( fmt::format("Failed to read OVF file \"{}\".", this->filename) );
        }
    }
    
    void OVF_File::Read_Data( vectorfield& vf )
    {
        try
        {
            auto lvl = Log_Level::Parameter;
            
            // Raw data representation
            myfile.Read_String( this->datatype_in, "# Begin: Data" );
            std::istringstream repr( this->datatype_in );
            repr >> this->datatype_in;
            if( this->datatype_in == "binary" ) 
                repr >> this->binary_length;
            
            Log( lvl, this->sender, fmt::format( "# OVF data representation = {}", this->datatype_in ) );
            Log( lvl, this->sender, fmt::format( "# OVF binary length       = {}", this->binary_length ) );
            
            // Check that representation and binary length valures are ok
            if( this->datatype_in != "text" && this->datatype_in != "binary" )
            {
                spirit_throw(Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    "Data representation must be either \"text\" or \"binary\"");
            }
            
            if( this->datatype_in == "binary" && 
                 this->binary_length != 4 && this->binary_length != 8  )
            {
                spirit_throw( Exception_Classifier::Bad_File_Content, Log_Level::Error,
                              "Binary representation can be either \"binary 8\" or \"binary 4\"");
            }
            
            // Read the data
            if( this->datatype_in == "binary" )
                Read_Data_bin( vf );
            else if( this->datatype_in == "text" )
                Read_Data_txt( vf ); 
        }
        catch (...) 
        {
            spirit_rethrow( fmt::format("Failed to read OVF file \"{}\".", filename) );
        }
    }
    
    void OVF_File::Read_Data_bin( vectorfield& vf )
    {
        try
        {        
            // Set the input stream indicator to the end of the line describing the data block
            myfile.iss.seekg( std::ios::end );
            
            // Check if the initial check value of the binary data is valid
            if( !Read_Check_Binary_Values() )
                spirit_throw( Exception_Classifier::Bad_File_Content, Log_Level::Error,
                              "The OVF initial binary value could not be read correctly");
            
            // Comparison of datum size compared to scalar type
            if ( this->binary_length == 4 )
            {
                int vectorsize = 3 * sizeof(float);
                float buffer[3];
                int index;
                for( int k=0; k<this->nodes[2]; k++ )
                {
                    for( int j=0; j<this->nodes[1]; j++ )
                    {
                        for( int i=0; i<this->nodes[0]; i++ )
                        {
                            index = i + j*this->nodes[0] + k*this->nodes[0]*this->nodes[1];
                            
                            myfile.myfile->read(reinterpret_cast<char *>(&buffer[0]), vectorsize);
                            
                            vf[index][0] = static_cast<scalar>(buffer[0]);
                            vf[index][1] = static_cast<scalar>(buffer[1]);
                            vf[index][2] = static_cast<scalar>(buffer[2]);
                        }
                    }
                }
                
            }
            else if (this->binary_length == 8)
            {
                int vectorsize = 3 * sizeof(double);
                double buffer[3];
                int index;
                for (int k = 0; k<this->nodes[2]; k++)
                {
                    for (int j = 0; j<this->nodes[1]; j++)
                    {
                        for (int i = 0; i<this->nodes[0]; i++)
                        {
                            index = i + j*this->nodes[0] + k*this->nodes[0] * this->nodes[1];
                            
                            myfile.myfile->read(reinterpret_cast<char *>(&buffer[0]), vectorsize);
                            
                            vf[index][0] = static_cast<scalar>(buffer[0]);
                            vf[index][1] = static_cast<scalar>(buffer[1]);
                            vf[index][2] = static_cast<scalar>(buffer[2]);
                        }
                    }
                }
            }
        }
        catch (...)
        {
            spirit_rethrow( "Failed to read OVF binary data" );
        }
    }

    void OVF_File::Read_Data_txt( vectorfield& vf )
    {
        try
        {   
            int nos = this->nodes[0] * this->nodes[1] * this->nodes[2];
            
            for (int i=0; i<nos; i++)
            {
                this->myfile.GetLine();
                this->myfile.iss >> vf[i][0];
                this->myfile.iss >> vf[i][1];
                this->myfile.iss >> vf[i][2];
            }
        }
        catch (...)
        {
            spirit_rethrow( "Failed to check OVF initial binary value" );
        }
    }
    
    bool OVF_File::Read_Check_Binary_Values()
    {
        try
        {
            // create initial check values for the binary data (see OVF specification)
            const double ref_8b = *reinterpret_cast<const double *>( &this->test_hex_8b );
            double read_8byte = 0;
            
            const float ref_4b = *reinterpret_cast<const float *>( &this->test_hex_4b );
            float read_4byte = 0;
            
            // check the validity of the initial check value read with the reference one
            if ( this->binary_length == 4 )
            {    
                myfile.myfile->read( reinterpret_cast<char *>( &read_4byte ), sizeof(float) );
                if ( read_4byte != ref_4b ) 
                {
                    spirit_throw( Exception_Classifier::Bad_File_Content, Log_Level::Error,
                                  "OVF initial check value of binary data is inconsistent" );
                }
            }
            else if ( this->binary_length == 8 )
            {
                myfile.myfile->read( reinterpret_cast<char *>( &read_8byte ), sizeof(double) );
                if ( read_8byte != ref_8b )
                {
                    spirit_throw( Exception_Classifier::Bad_File_Content, Log_Level::Error,
                                  "OVF initial check value of binary data is inconsistent" );
                }
            }
            
            return true;
        }
        catch (...)
        {
            spirit_rethrow( "Failed to check OVF initial binary value" );
            return false;
        }
    }

    // Public methods
    
    void OVF_File::write_image( const vectorfield& vf, const Data::Geometry& geometry )
    {
        Write_Top_Header(1);
        Write_Segment( vf, geometry );
    }

    void OVF_File::write_eigenmodes( const std::vector<std::shared_ptr<vectorfield>>& modes,
                                     const Data::Geometry& geometry )
    {
        Write_Top_Header( modes.size() );
        for (int i=0; i<modes.size(); i++)
        {
            if ( modes[i] != NULL )
                Write_Segment( *modes[i], geometry );
        }
    }

    void OVF_File::write_chain( const std::shared_ptr<Data::Spin_System_Chain>& chain )
    {
        Write_Top_Header( chain->noi );
        for (int i=0; i<chain->noi; i++)
            Write_Segment( *chain->images[i]->spins, *chain->images[i]->geometry );
    }

    
    void OVF_File::read_image( vectorfield& vf, Data::Geometry& geometry )
    {
        Read_Header();
        Read_Check_Geometry( geometry );
        Read_Data( vf );
    }

    void OVF_File::read_eigenmodes( std::vector<std::shared_ptr<vectorfield>>& modes,
                                    Data::Geometry& geometry )
    {
    }
}
