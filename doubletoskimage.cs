//Code Example: Convert double[] to SKImage
using SkiaSharp;
using System;
using System.IO;

class Program
{
    static void Main()
    {
        // Example: 1D array representing pixel values (grayscale) for an image
        double[] pixelData = new double[]
        {
            255, 200, 150, 100, 50, 0, 125, 255, 200, 100,
            255, 100, 50, 200, 0, 125, 255, 255, 200, 100
        };

        // Image dimensions (width and height of the image)
        int width = 5;  // width of the image
        int height = 4; // height of the image (the array should have width * height elements)

        // Create an SKBitmap to hold the image data
        using (SKBitmap bitmap = new SKBitmap(width, height))
        {
            // Ensure the pixelData length matches width * height
            if (pixelData.Length != width * height)
            {
                throw new ArgumentException("The pixel data array size does not match the image dimensions.");
            }

            // Convert the 1D array to the bitmap's pixel data
            int index = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Get the pixel value (convert to byte and clamp to 0-255)
                    byte pixelValue = (byte)Math.Clamp(pixelData[index], 0, 255);

                    // Set the pixel color (grayscale)
                    bitmap.SetPixel(x, y, new SKColor(pixelValue, pixelValue, pixelValue));

                    index++; // Move to the next element in the pixelData array
                }
            }

            // Encode the SKBitmap into an SKImage
            SKImage skImage = SKImage.FromBitmap(bitmap);

            // Optionally save the image to a file
            using (var stream = File.OpenWrite("output_image.png"))
            {
                SKData skData = skImage.Encode();
                skData.SaveTo(stream);
            }

            Console.WriteLine("Image created and saved as 'output_image.png'.");
        }
    }
}
