using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace Video_OpenCV
{
    class Program
    {
        static void Main()
        {
            // Declaración de variables y carga de video
            Console.WriteLine("Ingresar la dirección del video");
            string rutaVideo = Console.ReadLine();
            string carpetaSalida = @"C:\opencv\frames\";
            int contadorFrames = 0;
            VideoCapture video = new VideoCapture(rutaVideo);
            CascadeClassifier detector = new CascadeClassifier(@"C:\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml");
            Mat frameReferencia = null;
            bool primerFotograma = true;

            // Crear carpeta de salida si no existe
            if (!Directory.Exists(carpetaSalida))
            {
                Directory.CreateDirectory(carpetaSalida);
            }

            // Crear una ventana para mostrar los resultados
            CvInvoke.NamedWindow("Diferencias Faciales", NamedWindowType.Normal);

            // Leer cada fotograma hasta que el video termine
            while (true)
            {
                Mat frame = video.QueryFrame();

                if (frame == null || frame.IsEmpty)
                    break;

                // Convertir la imagen a escala de grises para la detección facial
                Mat frameGris = new Mat();
                CvInvoke.CvtColor(frame, frameGris, ColorConversion.Bgr2Gray);

                // Mostrar frame original
                CvInvoke.Imshow("Frame original", frame);

                // Detectar rostros en la imagen
                
                var rostros = detector.DetectMultiScale(frameGris, 1.3, 5);

                // Procesar cada rostro detectado
                foreach (var rostro in rostros)
                {
                    Mat regionRostro = new Mat(frameGris, rostro);

                    // Detectar sonrisas
                    CascadeClassifier clasificadorSonrisa = new CascadeClassifier(@"C:\opencv\build\etc\haarcascades\haarcascade_smile.xml");
                    var sonrisas = clasificadorSonrisa.DetectMultiScale(regionRostro, 1.8, 20);

                    foreach (var rectRostro in rostros)
                    {
                        CvInvoke.Rectangle(frame, rectRostro, new MCvScalar(0, 225, 0), 2);

                        foreach (var rectSonrisa in sonrisas)
                        {
                            CvInvoke.Rectangle(frame, new Rectangle(rectRostro.X + rectSonrisa.X, rectRostro.Y + rectSonrisa.Y,
                                rectSonrisa.Width, rectSonrisa.Height), new MCvScalar(0, 0, 255), 2);
                        }
                        frameReferencia = new Mat(regionRostro.Size, DepthType.Cv8U, 1);
                    }


                    //Verificar y convertir a escala de grises si es necesario
                    if (frameReferencia.NumberOfChannels != 1)
                    {
                        CvInvoke.CvtColor(frameReferencia, frameReferencia, ColorConversion.Bgr2Gray);
                    }
                    if (regionRostro.NumberOfChannels != 1)
                    {
                        CvInvoke.CvtColor(regionRostro, regionRostro, ColorConversion.Bgr2Gray);
                    }

                    // Comparar con la imagen de referencia o almacenar la referencia si aún no está establecida
                    if (primerFotograma)
                    {
                        regionRostro.CopyTo(frameReferencia);
                        primerFotograma = false;
                    }
                    else
                    {
                        // Calcular las diferencias faciales
                        Mat diferencias = new Mat();
                        CvInvoke.AbsDiff(frameReferencia, regionRostro, diferencias);

                        // Asegurarse de que las matrices tengan las mismas dimensiones y código
                        if (diferencias.Size != frameReferencia.Size)
                        {
                            CvInvoke.Resize(regionRostro, regionRostro, frameReferencia.Size);
                        }

                        // Mostrar la imagen con las diferencias
                        CvInvoke.Imshow("Diferencias Faciales", diferencias);
                    }
                }

                // Guardar frame
                string rutaImagen = Path.Combine(carpetaSalida, $"frame_{contadorFrames}.png");
                CvInvoke.Imwrite(rutaImagen, frame);

                contadorFrames++;
            }

            // Liberar los recursos
            video.Dispose();
            CvInvoke.DestroyAllWindows();

            Console.WriteLine($"Número de frames procesados: {contadorFrames}");
            Console.ReadKey();
        }
    }
}
