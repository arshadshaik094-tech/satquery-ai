import { Database, ExternalLink } from "lucide-react";
import { PageHeader } from "./SingleImageAnalysis";

const DATASETS = [
  {
    name: "EuroSAT",
    use: "Land-cover scene classification (10 classes) — fine-tune target for the ResNet-50 head",
    size: "27,000 labeled 64×64 Sentinel-2 patches",
    url: "https://github.com/phelber/EuroSAT",
  },
  {
    name: "LEVIR-CD",
    use: "Bi-temporal building change detection — fine-tune target for MobileNet-V3 change head",
    size: "637 image pairs, 0.5m resolution, Google Earth",
    url: "https://justchenhao.github.io/LEVIR/",
  },
  {
    name: "SEN12MS",
    use: "Co-registered Sentinel-1 SAR / Sentinel-2 optical patches — fine-tune target for dual ResNet-18 fusion head",
    size: "180,662 triplets across all seasons",
    url: "https://mediatum.ub.tum.de/1474000",
  },
  {
    name: "RSVQA-LR",
    use: "Remote-sensing VQA pairs (auto-generated from OpenStreetMap) — eval/fine-tune for BLIP-VQA",
    size: "772 Sentinel-2 tiles, ~77k Q/A pairs",
    url: "https://rsvqa.sylvainlobry.com/",
  },
  {
    name: "DOTA v2",
    use: "Oriented object detection in aerial imagery — optional specialist for 'major objects visible' queries",
    size: "11,268 images, 18 categories",
    url: "https://captain-whu.github.io/DOTA/index.html",
  },
  {
    name: "ISRO Bhuvan / VEDAS",
    use: "Indian national EO data products — scalability narrative & real imagery source",
    size: "National geoportal",
    url: "https://bhuvan.nrsc.gov.in",
  },
];

export default function Datasets() {
  return (
    <div className="p-6">
      <PageHeader
        title="Datasets"
        subtitle="Every dataset referenced is real and publicly accessible — links go to the official source, not a mirror."
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-5">
        {DATASETS.map((d) => (
          <div key={d.name} className="card p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Database size={15} className="text-brand-600" />
                <h3 className="font-semibold text-sm text-ink-900">{d.name}</h3>
              </div>
              <a href={d.url} target="_blank" rel="noreferrer" className="text-ink-400 hover:text-brand-600">
                <ExternalLink size={14} />
              </a>
            </div>
            <p className="text-xs text-ink-600 leading-relaxed">{d.use}</p>
            <p className="text-[11px] text-ink-400">{d.size}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
