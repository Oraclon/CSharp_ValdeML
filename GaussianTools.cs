namespace Ophanim.FiltersLIB
{
    public static class Gaussian
    {
        private static double _Clamp(this double pixel, double maxValue = 1)
        {
            return Math.Max(0, Math.Min(maxValue, pixel));
        }
        private static double _NextGaussian(double mean, double deviation)
        {
            Random _rnd = new Random();
            double u1 = 1.0 - _rnd.NextDouble();
            double u2 = 1.0 - _rnd.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + deviation * randStdNormal;
        }
        private static void _NextGaussianArray(this double[] array, double mean, double deviation)
        {
            int _features = array.Length;
            for (int feature = 0; feature < _features; feature++)
                array[feature] = (array[feature] + _NextGaussian(mean, deviation))._Clamp();
        }
        private static double[] _GenerateGaussianKernel(this int kernelSize, double deviation)
        {
            double minSigma = 0.8;
            double maxSigma = 2.0;
            Random rand = new Random();
            double std = deviation > 0 ? deviation : minSigma + rand.NextDouble() * (maxSigma - minSigma);
            double[] generatedKernel = new double[(int)Math.Pow(kernelSize, 2)];
            double sum = 0;

            int half = kernelSize / 2;
            int pointer = 0;

            for (int y = -half; y <= half; y++)
                for (int x = -half; x <= half; x++)
                {
                    double exponent = -(Math.Pow(x, 2) + Math.Pow(y, 2)) / (2 * Math.Pow(std, 2));
                    double value = (1 / (2 * Math.PI * Math.Pow(std, 2)) * Math.Exp(exponent));
                    generatedKernel[pointer] = value;
                    sum += value;
                    pointer++;
                }

            //Normalize Kernel
            for (int feature = 0; feature < generatedKernel.Length; feature++)
                generatedKernel[feature] /= sum;

            return generatedKernel;
        }

        public static double[][][] Noise(this double[][][] samples, double mean = 0, double deviation = 1.0)
        {
            double[][][] clone = (double[][][])samples.Clone();

            for (int sample = 0; sample < samples.Length; sample++)
            {
                int channels = samples[sample].Length;
                Parallel.For(0, channels, channel => {
                    clone[sample][channel]._NextGaussianArray(mean, deviation);
                });
            }
            return clone;
        }
        public static double[][][] Blur(this double[][][] samples, int kernelSize, double deviation = 0)
        {
            double[] kernelValues = kernelSize._GenerateGaussianKernel(deviation);

            double[][][] clone = (double[][][])samples.Clone();
            for (int sample = 0; sample < samples.Length; sample++)
            {
            }
            return clone;
        }
    }
}
