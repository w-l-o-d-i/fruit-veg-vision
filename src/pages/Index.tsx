import { useState, useEffect } from "react";
import { Scale, Loader2 } from "lucide-react";
import { CameraCapture } from "@/components/CameraCapture";
import { ResultDisplay } from "@/components/ResultDisplay";
import { loadModel, classifyImage } from "@/lib/classifier";
import { fruitWeights } from "@/data/weights";
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    const initModel = async () => {
      try {
        await loadModel();
        setIsModelLoading(false);
      } catch (error) {
        console.error("Failed to load model:", error);
        toast({
          title: "Błąd ładowania modelu",
          description: "Nie udało się załadować modelu AI. Odśwież stronę.",
          variant: "destructive",
        });
      }
    };
    initModel();
  }, []);

  const handleCapture = async (imageData: string) => {
    setCapturedImage(imageData);
    setIsProcessing(true);

    try {
      const result = await classifyImage(imageData);
      setPrediction(result);
    } catch (error) {
      console.error("Classification error:", error);
      toast({
        title: "Błąd rozpoznawania",
        description: "Nie udało się rozpoznać owocu/warzywa. Spróbuj ponownie.",
        variant: "destructive",
      });
      setCapturedImage(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setCapturedImage(null);
    setPrediction(null);
  };

  if (isModelLoading) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
        <div className="text-center space-y-4">
          <div className="h-16 w-16 mx-auto rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center animate-pulse">
            <Scale className="h-8 w-8 text-primary-foreground" />
          </div>
          <h1 className="text-2xl font-bold">Inteligentna Waga</h1>
          <p className="text-muted-foreground flex items-center gap-2 justify-center">
            <Loader2 className="h-4 w-4 animate-spin" />
            Ładowanie modelu AI...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center">
              <Scale className="h-6 w-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Inteligentna Waga
            </h1>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {isProcessing ? (
          <div className="max-w-2xl mx-auto flex flex-col items-center justify-center space-y-4 py-20">
            <Loader2 className="h-12 w-12 animate-spin text-primary" />
            <p className="text-lg text-muted-foreground">Rozpoznawanie...</p>
          </div>
        ) : capturedImage && prediction ? (
          <ResultDisplay
            image={capturedImage}
            prediction={prediction}
            weightData={fruitWeights[prediction] || { name: prediction, min: 0, max: 0, avg: 0 }}
            onReset={handleReset}
          />
        ) : (
          <div className="max-w-4xl mx-auto">
            <div className="mb-6 text-center">
              <h2 className="text-3xl font-bold mb-2">Zrób zdjęcie owocu lub warzywa</h2>
              <p className="text-muted-foreground">
                Użyj aparatu, aby rozpoznać produkt i poznać jego wagę
              </p>
            </div>
            <div className="aspect-video rounded-2xl overflow-hidden shadow-xl">
              <CameraCapture onCapture={handleCapture} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Index;
