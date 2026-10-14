export default function StatCard({ label, value, delta, deltaGood = true, icon: Icon }) {
  return (
    <div className="card p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-ink-500">{label}</span>
        {Icon && (
          <div className="w-7 h-7 rounded-lg bg-brand-50 grid place-items-center text-brand-600">
            <Icon size={14} />
          </div>
        )}
      </div>
      <span className="text-[26px] font-bold text-ink-900 leading-none tabular-nums">{value}</span>
      {delta && (
        <span className={`text-xs font-medium ${deltaGood ? "text-accent-teal" : "text-accent-red"}`}>
          {delta}
        </span>
      )}
    </div>
  );
}
