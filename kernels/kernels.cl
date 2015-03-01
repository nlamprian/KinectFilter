kernel
void convolution ( read_only image2d_t sourceImage,
                   write_only image2d_t outputImage,
                   uint rows, uint cols,
                   constant float *filter,
                   uint filterWidth,
                   sampler_t sampler )
{
    // Store each work-item's unique row column
    uint column = get_global_id (0);
    uint row = get_global_id (1);

    // Half the width of the filter is needed for indexing memory later
    int halfwidth = filterWidth / 2;

    // All accesses to images return data as four-element vector
    // (i.e. float4), although only the 'x' component will contain
    // meaningful data in this code
    // float4 sum = { 0.0f, 0.0f, 0.0f, 0.0f };
    float sum = 0;

    // Iterator for the filter
    int filterIdx = 0;

    // Each work-item iterates around its local area based
    // on the size of the filter
    int2 coords; // Coordinates for accessing the image
    // Iterate over the filter rows
    for (int i = -halfwidth; i <= halfwidth; ++i)
    {
        coords.y = row + i;

        // Iterate over the filter columns
        for (int j = -halfwidth; j <= halfwidth; ++j)
        {
            coords.x = column + j;

            uint4 pixel;
            // Read a pixel from the image. A single channel image
            // stores the pixel in the 'x' coordinate of the returned vector
            pixel = read_imageui (sourceImage, sampler, coords);
            sum += pixel.x * filter[filterIdx++];
        }
    }

    // Copy the data to the output image,
    // if the work-item is within bounds
    if (row < rows && column < cols)
    {
        coords.x = column;
        coords.y = row;
        uint4 color = { sum, 0, 0, 0 };
        write_imageui (outputImage, coords, color);
    }
}


kernel
void convolutionGL ( read_only image2d_t sourceImage,
                     write_only image2d_t outputImage,
                     uint rows, uint cols,
                     constant float *filter,
                     uint filterWidth,
                     sampler_t sampler )
{
    // Store each work-item's unique row column
    uint column = get_global_id (0);
    uint row = get_global_id (1);

    // Half the width of the filter is needed for indexing memory later
    int halfwidth = filterWidth / 2;

    // All accesses to images return data as four-element vector
    // (i.e. float4), although only the 'x' component will contain
    // meaningful data in this code
    // float4 sum = { 0.0f, 0.0f, 0.0f, 0.0f };
    float sum = 0;

    // Iterator for the filter
    int filterIdx = 0;

    // Each work-item iterates around its local area based
    // on the size of the filter
    int2 coords; // Coordinates for accessing the image
    // Iterate over the filter rows
    for (int i = -halfwidth; i <= halfwidth; ++i)
    {
        coords.y = row + i;

        // Iterate over the filter columns
        for (int j = -halfwidth; j <= halfwidth; ++j)
        {
            coords.x = column + j;

            float4 pixel;
            // Read a pixel from the image. A single channel image
            // stores the pixel in the 'x' coordinate of the returned vector
            pixel = read_imagef (sourceImage, sampler, coords);
            sum += pixel.x * filter[filterIdx++];
        }
    }

    // Copy the data to the output image,
    // if the work-item is within bounds
    if (row < rows && column < cols)
    {
        coords.x = column;
        coords.y = row;
        float4 color = { sum, sum, sum, 1.f };
        write_imagef (outputImage, coords, color);
    }
}


kernel
void normalizeImg ( read_only image2d_t sourceImage,
                    write_only image2d_t outputImage,
                    uint rows, uint cols,
                    sampler_t sampler )
{
    // Store each work-item's unique row column
    uint column = get_global_id (0);
    uint row = get_global_id (1);

    // Coordinates for accessing the image
    int2 coords = (int2) (column, row);

    uint4 pixel = read_imageui (sourceImage, sampler, coords);

    // Copy the data to the output image,
    // if the work-item is within bounds
    if (row < rows && column < cols)
    {
        float4 pixelf = convert_float4 (pixel) / 255.f;
        pixelf.w = 1.f;
        write_imagef (outputImage, coords, pixelf);
    }
}


kernel
void rgb2rgba ( global uchar *rgb, global float4 *rgba,
                uint rows, uint cols )
{
    // Store each work-item's unique row column
    uint column = get_global_id (0);
    uint row = get_global_id (1);

    // Flatten indices
    uint idx = row * cols + column;

    // Read pixel in
    uchar3 pixel = vload3 (idx, rgb);

    // Copy the data to the output image,
    // if the work-item is within bounds
    if (row < rows && column < cols)
    {
        rgba[idx] = (float4) (convert_float3 (pixel) / 255.f, 1.f);
    }
}


kernel
void normalizeRGB ( global float4 *in, global float4 *out,
                    uint rows, uint cols )
{
    // Store each work-item's unique row column
    uint column = get_global_id (0);
    uint row = get_global_id (1);

    // Flatten indices
    uint idx = row * cols + column;

    // Calculate normalizing factor
    float4 pixel = in[idx];
    float sum = pixel.x + pixel.y + pixel.z;
    
    // Copy the data to the output image,
    // if the work-item is within bounds
    if (row < rows && column < cols)
    {
        // Normalize and store
        pixel /= sum;
        pixel.w = 1.f;
        out[idx] = pixel;
    }
}


kernel
void depthTo3D (global ushort *depth, global float4 *pCloud, float f)
{
    // Workspace dimensions
    uint cols = get_global_size (0);
    uint rows = get_global_size (1);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    // Flatten indices
    uint idx = gY * cols + gX;

    float d = convert_float (depth[idx]);
    float4 point = { (gX - (cols - 1) / 2.f) * d / f,  // X = (x - cx) * d / fx
                     (gY - (rows - 1) / 2.f) * d / f,  // Y = (y - cy) * d / fy
                     d, 1.f };                         // Z = d

    pCloud[idx] = point;
}
