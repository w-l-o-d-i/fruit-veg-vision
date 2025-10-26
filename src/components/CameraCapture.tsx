import { useRef, useState, useEffect } from "react";
import { Camera, SwitchCamera } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

interface CameraCaptureProps {
  onCapture: (imageData: string) => void;
}

export const CameraCapture = ({ onCapture }: CameraCaptureProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment");
  const { toast } = useToast();

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      toast({
        title: "Błąd aparatu",
        description: "Nie można uzyskać dostępu do aparatu. Sprawdź uprawnienia.",
        variant: "destructive",
      });
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      setIsStreaming(false);
    }
  };

  const switchCamera = async () => {
    stopCamera();
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      if (context) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL("image/jpeg", 0.9);
        onCapture(imageData);
        stopCamera();
      }
    }
  };

  useEffect(() => {
    if (facingMode) {
      startCamera();
    }
    return () => stopCamera();
  }, [facingMode]);

  return (
    <div className="relative w-full h-full flex flex-col items-center justify-center">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="w-full h-full object-cover rounded-2xl"
      />
      <canvas ref={canvasRef} className="hidden" />

      {isStreaming && (
        <div className="absolute bottom-6 left-0 right-0 flex justify-center gap-4 px-4">
          <Button
            variant="secondary"
            size="icon"
            onClick={switchCamera}
            className="h-14 w-14 rounded-full bg-card/90 backdrop-blur-sm hover:bg-card"
          >
            <SwitchCamera className="h-6 w-6" />
          </Button>
          <Button
            onClick={capturePhoto}
            size="lg"
            className="h-16 w-16 rounded-full shadow-lg"
          >
            <Camera className="h-8 w-8" />
          </Button>
        </div>
      )}
    </div>
  );
};
