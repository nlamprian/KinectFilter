__kernel
void convolution( __read_only image2d_t sourceImage,
                  __write_only image2d_t outputImage,
                  int rows, int cols,
                  __constant float* filter,
                  int filterWidth,
                  sampler_t sampler )
{
    // Store each work-item's unique row column
    int column = get_global_id(0);
    int row = get_global_id(1);

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
            pixel = read_imageui(sourceImage, sampler, coords);
            sum += pixel.x * filter[filterIdx++];
        }
    }

    // Copy the data to the output image if the
    // work-item is in bounds
    if (row < rows && column < cols)
    {
        coords.x = column;
        coords.y = row;
        uint4 color = { (uint)sum, 0, 0, 0 };
        write_imageui(outputImage, coords, color);
    }
}
