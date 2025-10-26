import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Camera, Scale } from "lucide-react";
import { FruitWeight } from "@/data/weights";

interface ResultDisplayProps {
  image: string;
  prediction: string;
  weightData: FruitWeight;
  onReset: () => void;
}

export const ResultDisplay = ({ image, prediction, weightData, onReset }: ResultDisplayProps) => {
  return (
    <div className="w-full max-w-2xl mx-auto space-y-6 animate-in fade-in duration-500">
      <Card className="overflow-hidden">
        <img
          src={image}
          alt="Captured fruit"
          className="w-full h-64 object-cover"
        />
      </Card>

      <Card className="p-8 space-y-6">
        <div className="flex items-center gap-3">
          <div className="h-12 w-12 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center">
            <Scale className="h-6 w-6 text-primary-foreground" />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Rozpoznano</p>
            <h2 className="text-2xl font-bold">1x {prediction}</h2>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-4 rounded-xl bg-muted/50">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Min</p>
            <p className="text-2xl font-bold text-foreground">{weightData.min}g</p>
          </div>
          <div className="text-center p-4 rounded-xl bg-gradient-to-br from-primary/10 to-accent/10 border-2 border-primary/20">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Średnia</p>
            <p className="text-3xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              {weightData.avg}g
            </p>
          </div>
          <div className="text-center p-4 rounded-xl bg-muted/50">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Max</p>
            <p className="text-2xl font-bold text-foreground">{weightData.max}g</p>
          </div>
        </div>
      </Card>

      <Button
        onClick={onReset}
        variant="outline"
        className="w-full h-14 text-lg"
      >
        <Camera className="mr-2 h-5 w-5" />
        Zrób nowe zdjęcie
      </Button>
    </div>
  );
};
