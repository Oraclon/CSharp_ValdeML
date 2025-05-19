//Code Example: Converting a double[] Array to SKImage for RGB
using SkiaSharp;
using System;
using System.IO;

class Program
{
    static void Main()
    {
        // Example: 1D array representing RGB pixel values for an image
        // Each set of three consecutive values represents R, G, B values for a pixel
        double[] pixelData = new double[]
        {
            255, 0, 0,   // Red pixel
            0, 255, 0,   // Green pixel
            0, 0, 255,   // Blue pixel
            255, 255, 0, // Yellow pixel
            0, 255, 255, // Cyan pixel
            255, 0, 255  // Magenta pixel
        };

        // Image dimensions (width and height of the image)
        int width = 2;  // width of the image (in this case, 2 pixels per row)
        int height = 3; // height of the image (3 rows)

        // Create an SKBitmap to hold the image data
        using (SKBitmap bitmap = new SKBitmap(width, height))
        {
            // Ensure the pixelData length matches width * height * 3 (3 channels for RGB)
            if (pixelData.Length != width * height * 3)
            {
                throw new ArgumentException("The pixel data array size does not match the image dimensions.");
            }

            // Convert the 1D array to the bitmap's pixel data
            int index = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Extract the RGB values from the array
                    byte red = (byte)Math.Clamp(pixelData[index++], 0, 255);
                    byte green = (byte)Math.Clamp(pixelData[index++], 0, 255);
                    byte blue = (byte)Math.Clamp(pixelData[index++], 0, 255);

                    // Set the pixel color (RGB)
                    bitmap.SetPixel(x, y, new SKColor(red, green, blue));
                }
            }

            // Encode the SKBitmap into an SKImage
            SKImage skImage = SKImage.FromBitmap(bitmap);

            // Optionally save the image to a file
            using (var stream = File.OpenWrite("rgb_image.png"))
            {
                SKData skData = skImage.Encode();
                skData.SaveTo(stream);
            }

            Console.WriteLine("RGB Image created and saved as 'rgb_image.png'.");
        }
    }
}
