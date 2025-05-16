using System;

class GaussianNoiseGenerator
{
    private Random random = new Random();
    private bool hasSpare = false;
    private double spare;

    public double NextGaussian(double mu = 0.0, double sigma = 1.0)
    {
        if (hasSpare)
        {
            hasSpare = false;
            return spare * sigma + mu;
        }

        double u, v, s;

        do
        {
            u = 2.0 * random.NextDouble() - 1.0;
            v = 2.0 * random.NextDouble() - 1.0;
            s = u * u + v * v;
        }
        while (s >= 1.0 || s == 0.0);

        s = Math.Sqrt(-2.0 * Math.Log(s) / s);

        spare = v * s;
        hasSpare = true;

        return mu + sigma * u * s;
    }
}

class AddGaussianNoiseToImage
{
    static int Clamp(double value, int min, int max)
    {
        return (int)Math.Max(min, Math.Min(max, Math.Round(value)));
    }

    static void Main()
    {
        int[,] image = new int[3, 3]
        {
            { 100, 120, 130 },
            { 125, 135, 140 },
            { 110, 115, 125 }
        };

        var noiseGen = new GaussianNoiseGenerator();
        double mu = 0.0;
        double sigma = 10.0;

        Console.WriteLine("Original Image:");
        PrintImage(image);

        Console.WriteLine("\nNoisy Image:");
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double noisy = image[i, j] + noiseGen.NextGaussian(mu, sigma);
                image[i, j] = Clamp(noisy, 0, 255);
            }
        }

        PrintImage(image);
    }

    static void PrintImage(int[,] image)
    {
        for (int i = 0; i < image.GetLength(0); i++)
        {
            for (int j = 0; j < image.GetLength(1); j++)
            {
                Console.Write($"{image[i, j],4}");
            }
            Console.WriteLine();
        }
    }
}
