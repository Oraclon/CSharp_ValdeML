using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DevelopmentTests
{
	public static class FastLoop
	{
		public static void Run(int maxRange)
		{
            var random = new Random(400);
            var items = Enumerable.Range(0, maxRange).Select(x => random.Next()).ToArray();
            Span<int> listAsSpan = items;
            ref var searchspace = ref MemoryMarshal.GetReference(listAsSpan);
            for (var i = 0; i < listAsSpan.Length; i++)
            {
                var item = Unsafe.Add(ref searchspace, i);
                Console.WriteLine($"{i} {item}");
            }
        }
	}
}

