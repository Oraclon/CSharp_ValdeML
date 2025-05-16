using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Xml.Linq;
using Newtonsoft.Json;
using OpenCL.Net;
public static class ImgProcessingTools
{
    public static double Clamp(this double value, double min, double max)
    {
        return Math.Max(min, Math.Min(max, value));
    }
    public static double[] ApplyGaussianNoiseOnImage(this double[] pixels, double mu = 0, double sigma = 10)
    {
        GaussianNoiseGenerator gaussian = new GaussianNoiseGenerator();
        double min = pixels.Min();
        double max = pixels.Max();

        for (int pixel = 0; pixel < pixels.Length; pixel++)
        {
            double noisy = pixels[pixel] + gaussian.NextGaussian(mu, sigma);
            pixels[pixel] = noisy.Clamp(0, 1);
        }

        return pixels;
    }
}
public class GaussianNoiseGenerator
{
    private Random _Random = new Random();
    public bool HasSpare = false;
    private double _Spare = 0;
    //--Sigma
    //Small sigma(e.g., sigma = 0.01):

    //The noise added to the image will be very small.

    //The image will have barely noticeable noise.

    //The pixel values will stay very close to the original values with only tiny random variations.

    //Larger sigma (e.g., sigma = 0.1 or sigma = 0.2):

    //The noise will become more pronounced.

    //The image will show stronger noise with noticeable random variations in pixel values.

    //--MU
    //Case 1: mu = 0.0 (Zero Mean)
    //If you add noise with mu = 0.0, the noise will be centered around zero, meaning:

    //Some pixels will increase in value(brighten).

    //Some pixels will decrease in value(darken).

    //So, the image will have random variations around the original pixel values.

    //Case 2: mu = 0.1 (Positive Mean)
    //If you add noise with mu = 0.1, the noise will on average increase the pixel values by 0.1:

    //The image will tend to brighten.

    //Some pixels will still decrease slightly (due to the randomness of the noise), but overall, most pixels will become slightly brighter.

    //Case 3: mu = -0.1 (Negative Mean)
    //If you add noise with mu = -0.1, the noise will on average decrease the pixel values by 0.1:

    //The image will tend to darken.

    //Most pixels will become darker, but some will still increase slightly due to randomness.

    public double NextGaussian(double mu = 0, double sigma = 1.0)
    {
        if (HasSpare)
        {
            HasSpare = false;
            return _Spare * sigma + mu;
        }

        double u = 0;
        double v = 0;
        double s = 0;

        do
        {
            u = 2 * _Random.NextDouble() - 1;
            v = 2 * _Random.NextDouble() - 1;
            s = u * u + v * v;
        }
        while (s >= 1.0 || s == 0);

        s = Math.Sqrt(-2 * Math.Log(s) / s);
        _Spare = v * s;
        HasSpare = true;

        return mu + sigma * u * s;
    }
}
class Programaa
{
    static void Main()
    {
        Random rand = new Random();

        double[] pixels = new double[]{ 0.39, 0.47, 0.51, 0.49, 0.53, 0.55, 0.43, 0.45, 0.49 };
        pixels.ApplyGaussianNoiseOnImage(0, .5);
    }
}
