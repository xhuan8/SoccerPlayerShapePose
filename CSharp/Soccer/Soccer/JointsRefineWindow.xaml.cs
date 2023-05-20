using LCommon.Common;
using LCommon.Vision;
using LCommon.Vision.Diagnostics;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace Soccer
{
    /// <summary>
    /// JointsRefineWindow.xaml 的交互逻辑
    /// </summary>
    public partial class JointsRefineWindow : Window
    {
        private List<string> images = new List<string>();
        private List<string> data = new List<string>();
        private List<double> scoreList = new List<double>();
        private List<string> result_images = new List<string>();
        private int current = -1;
        private LImage currentImage;
        private LImage resultImage;
        private List<OverlayText> overlayText = new List<OverlayText>();
        private string jointFolder;
        private string scoreFolder;
        private string imageFolder;
        private float maxScore;
        private string currentGame;
        private List<string> refined;
        private string refinedFile = "Refined.xml";

        public JointsRefineWindow()
        {
            InitializeComponent();
        }

        public void Initialize(string jointFolder, string scoreFolder, string imageFolder, float maxScore,
            string refinedFile = "Refined.xml")
        {
            this.jointFolder = jointFolder;
            this.scoreFolder = scoreFolder;
            this.imageFolder = imageFolder;
            this.maxScore = maxScore;
            this.refinedFile = refinedFile;
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            string folder = scoreFolder;
            if (string.IsNullOrEmpty(folder))
                folder = jointFolder;
            refined = new List<string>();
            if (File.Exists(refinedFile))
                refined = SerializeHelper.LoadXML(refinedFile, typeof(List<string>)) as List<string>;
            foreach (string game in Directory.GetDirectories(folder))
            {
                string gameName = System.IO.Path.GetFileName(game);
                if (refined.Contains(gameName))
                    continue;

                string gameJoint = System.IO.Path.Combine(jointFolder, System.IO.Path.GetFileName(game));
                string gameImage = System.IO.Path.Combine(imageFolder, System.IO.Path.GetFileName(game));
                DirectoryInfo info = new DirectoryInfo(game);
                var scenes = info.GetDirectories();
                Array.Sort(scenes, (x, y) => (Convert.ToInt32(x.Name).CompareTo(Convert.ToInt32(y.Name))));
                foreach (var sceneInfo in scenes)
                {
                    var scene = sceneInfo.FullName;
                    //if (int.Parse(System.IO.Path.GetFileName(scene)) < 108)
                    //    continue;
                    string sceneJoint = System.IO.Path.Combine(gameJoint, System.IO.Path.GetFileName(scene));
                    string sceneImage = System.IO.Path.Combine(gameImage, System.IO.Path.GetFileName(scene));

                    foreach (string player in Directory.GetDirectories(scene))
                    {
                        string playerJoint = System.IO.Path.Combine(sceneJoint, System.IO.Path.GetFileName(player));
                        string playerImage = System.IO.Path.Combine(sceneImage, System.IO.Path.GetFileName(player));

                        string scoreFile = System.IO.Path.Combine(player, "metrics.xml");

                        double score = 100;
                        if (File.Exists(scoreFile))
                        {
                            string json = File.ReadAllText(scoreFile);
                            score = (double)(JsonConvert.DeserializeObject(json) as JArray)[1];
                        }
                        //if (score > maxScore)
                        {
                            foreach (string view in Directory.GetFiles(playerImage))
                            {
                                string resultFile = System.IO.Path.Combine(player,
                                    System.IO.Path.GetFileName(view).Replace(".png", "_2.png"));
                                images.Add(view);
                                scoreList.Add(score);
                                result_images.Add(resultFile);

                                string dataFile = System.IO.Path.Combine(playerJoint, 
                                    System.IO.Path.GetFileName(view).Replace(".png", "_j2d.xml"));
                                data.Add(dataFile);
                            }
                        }
                    }
                }

                currentGame = gameName;
                break;
            }
        }

        private void btnPrevious_Click(object sender, RoutedEventArgs e)
        {
            current -= 1;
            LoadData();
        }
        private void btnNext_Click(object sender, RoutedEventArgs e)
        {
            current += 1;
            LoadData();
        }

        private void LoadData()
        {
            if (current >= 0 && current < images.Count)
            {
                if (currentImage != null)
                    currentImage.Dispose();
                currentImage = new LImage();
                currentImage.CreateFromFile(images[current]);
                imageDisplay.Image = currentImage;

                if (resultImage != null)
                    resultImage.Dispose();
                resultImage = new LImage();
                resultImage.CreateFromFile(result_images[current]);
                imageResult.Image = resultImage;

                txtImageFilename.Text = images[current];
                txtScore.Text = scoreList[current].ToString();
                txtIndex.Text = String.Format("{0} / {1}", current, images.Count);

                string json = File.ReadAllText(data[current]);
                var joints = JsonConvert.DeserializeObject(json) as JArray;
                List<BaseROI> rois = new List<BaseROI>();
                overlayText = new List<OverlayText>();

                for (int i = 0; i < joints.Count; i++)
                {
                    var box = joints[i];
                    int x = (int)box[0], y = (int)box[1];

                    CircleROI circle = new CircleROI(x, y, 10);
                    rois.Add(circle);

                    overlayText.Add(new OverlayText(x, y - 10 >= 0 ? y - 10 : y, i.ToString()));
                }
                imageDisplay.ROIList = rois;

                RebuildOverlay();
            }
            else
            {
                refined.Add(currentGame);
                SerializeHelper.SaveXML(refined, refinedFile);

                imageDisplay.Image = null;
                txtImageFilename.Text = string.Empty;
                txtScore.Text = string.Empty;
                imageResult.Image = null;
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (currentImage != null)
                currentImage.Dispose();

            if (resultImage != null)
                resultImage.Dispose();
        }

        private void btnSave_Click(object sender, RoutedEventArgs e)
        {
            string filename = data[current];

            List<List<int>> lists = new List<List<int>>();
            foreach (BaseROI roi in imageDisplay.ROIList)
            {
                List<int> list = new List<int>();
                CircleROI circle = roi as CircleROI;
                list.Add((int)circle.Center.X);
                list.Add((int)circle.Center.Y);
                lists.Add(list);
            }
            string json = JsonConvert.SerializeObject(lists);
            File.WriteAllText(filename, json);

            RebuildOverlay();
        }

        private void RebuildOverlay()
        {
            Overlay overlay = new Overlay();
            for (int i = 0; i < overlayText.Count; i++)
            {
                overlayText[i].Pos = new Point(imageDisplay.ROIList[i].Left, imageDisplay.ROIList[i].Top);
                overlayText[i].Color = Colors.White;
                overlayText[i].FontSize = 24;
                overlay.AddObject(overlayText[i]);
            }
            imageDisplay.Overlay = overlay;
        }
    }
}
