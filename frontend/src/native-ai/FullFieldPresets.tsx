/**
 * fillEveryField — populate every field (required and optional) from a set of
 * domain presets, matching the workspace "fill all fields" standard.
 */
export function fillEveryField(
  values: Record<string, unknown>,
  variant = 0,
  choices: Record<string, string[]> = {}
): Record<string, unknown> {
  const names = ['Concise review', 'Team review', 'Evidence review'];
  return Object.fromEntries(
    Object.entries(values).map(([key, value]) => {
      if (['id', 'created_at', 'updated_at', 'user_id'].includes(key)) return [key, value];
      if (choices[key]) return [key, choices[key][variant % choices[key].length]];
      if (Array.isArray(value)) return [key, [`${names[variant]} example`]];
      if (value && typeof value === 'object') return [key, {}];
      if (typeof value === 'boolean') return [key, true];
      if (typeof value === 'number') return [key, [3, 2, 5][variant]];
      return [key, `${names[variant]}: review a deidentified workflow before any operational action. Identify evidence gaps and the human approvals needed. Do not invent records, measurements, completed integrations or approvals.`];
    })
  );
}

type Preset = { label: string; values: Record<string, unknown> };

type Props = {
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  choices?: Record<string, string[]>;
  presets?: Preset[];
  disabled?: boolean;
};

export default function FullFieldPresets({ value, onChange, choices = {}, presets, disabled = false }: Props) {
  const labels = presets ? presets.map((preset) => preset.label) : ['Concise review', 'Team review', 'Evidence review'];
  return (
    <div className="fill-every-field">
      <p><strong>Fill every field</strong> — each example fills required and optional fields. Review it before generating.</p>
      <div className="fill-every-field-actions">
        {labels.map((label, index) => (
          <button
            type="button"
            key={label}
            disabled={disabled}
            onClick={() => onChange(presets ? { ...value, ...presets[index].values } : fillEveryField(value, index, choices))}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
