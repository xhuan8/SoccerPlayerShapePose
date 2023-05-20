using LCommon.Common;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Soccer
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void button_Click(object sender, RoutedEventArgs e)
        {
            PlayerCropWindow window = new PlayerCropWindow();
            window.Initialize(@"E:\DataSet\FIFA", @"E:\Code\Soccer\Data\PlayerBroad");
            window.Show();
        }

        private void btnJointRefine_Click(object sender, RoutedEventArgs e)
        {
            JointsRefineWindow window = new JointsRefineWindow();
            window.Initialize(@"E:\Code\Soccer\Data\PlayerProxy", @"E:\Code\Soccer\Data\PlayerProxy",
                @"E:\Code\Soccer\Data\PlayerCrop", 10);
            window.Show();
        }

        private void btnJointRefineBroad_Click(object sender, RoutedEventArgs e)
        {
            JointsRefineWindowBroad window = new JointsRefineWindowBroad();
            window.Initialize(@"E:\Code\Soccer\Data\PlayerBroadProxy", @"E:\Code\Soccer\Data\PlayerCrop",
                @"E:\Code\Soccer\Data\PlayerBroadImage", 10);
            window.Show();
        }

        private void btnRealCrop_Click(object sender, RoutedEventArgs e)
        {
            PlayerCropWindow window = new PlayerCropWindow();
            window.Initialize(@"E:\Code\Soccer\Data\RealImages", @"E:\Code\Soccer\Data\RealPlayer", false);
            window.Show();
        }

        // 65
        private void btnRealJointRefine_Click(object sender, RoutedEventArgs e)
        {
            JointsRefineWindow window = new JointsRefineWindow();
            window.Initialize(@"E:\Code\Soccer\Data\RealPlayerProxy", null,
                @"E:\Code\Soccer\Data\RealPlayerImage", 10, "Refined_real.xml");
            window.Show();
        }

        private void btnTextureJointRefine_Click(object sender, RoutedEventArgs e)
        {
            JointsRefineWindow window = new JointsRefineWindow();
            window.Initialize(@"E:\Code\Soccer\Data\TextureProxy", null,
                @"E:\Code\Soccer\Data\TextureCrop", 10);
            window.Show();
        }

        private void btnSpliteRefined_Click(object sender, RoutedEventArgs e)
        {
            string folder = @"E:\Code\Soccer\Data\PlayerBroadImage";
            List<string> refined = new List<string>();
            string refinedFile = "Refined_Broad.xml";
            if (File.Exists(refinedFile))
                refined = SerializeHelper.LoadXML(refinedFile, typeof(List<string>)) as List<string>;
            int total = 0;
            foreach (string game in Directory.GetDirectories(folder))
            {
                string gameName = System.IO.Path.GetFileName(game);
                if (refined.Contains(gameName))
                    continue;

                foreach (string scene in Directory.GetDirectories(game))
                {
                    //if (int.Parse(System.IO.Path.GetFileName(scene)) < 61)
                    //    continue;
                    foreach (string player in Directory.GetDirectories(scene))
                    {
                        {
                            foreach (string view in Directory.GetFiles(player))
                            {
                                total++;
                            }
                        }
                    }
                }
            }
            MessageBox.Show(total.ToString());
        }
    }
}
