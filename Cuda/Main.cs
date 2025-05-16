using System;
using System.Runtime.InteropServices;

class Program
{
    const int rows = 3;
    const int cols = 3;
    //For LINUX -> [DllImport("libwrapper.so", CallingConvention = CallingConvention.Cdecl)]
    [DllImport("wrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void LaunchKernel(float[] data, int rows, int cols);

    static void Main()
    {
        float[] flatImage = new float[rows * cols]
        {
            0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f,
            0.7f, 0.8f, 0.9f
        };

        Console.WriteLine("Original Image:");
        Print(flatImage);

        LaunchKernel(flatImage, rows, cols);

        Console.WriteLine("\nImage After CUDA Kernel (each +1):");
        Print(flatImage);
    }

    static void Print(float[] data)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                Console.Write($"{data[i * cols + j]:0.00} ");
            }
            Console.WriteLine();
        }
    }
}

//nvcc -o wrapper.dll --shared wrapper.cpp kernel.cu
//FOR LINUX 
//nvcc -o libwrapper.so --shared wrapper.cpp kernel.cu

//If you want your C# program to be cross-platform, use RuntimeInformation to select the correct library dynamically:

//static string GetLibName()
//{
//    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
//        return "wrapper.dll";
//    else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
//        return "libwrapper.so";
//    else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
//        return "libwrapper.dylib";
//    else
//        throw new PlatformNotSupportedException();
//}

//[DllImport("__Internal", CallingConvention = CallingConvention.Cdecl)]
//static extern void LaunchKernel(...);
//Note: You may need to use NativeLibrary.SetDllImportResolver in .NET 6+ to redirect DllImport to your chosen file name dynamically.
