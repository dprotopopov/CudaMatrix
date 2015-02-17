using System;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CudaMatrix.UnitTest
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            string line = "-20890;14574;13811;-30854;3445;-26290;17281;-19563;17715;-15039;-28128;13352";
            Console.WriteLine(
                string.Join(" ", line.Split(';')));
        }
    }
}