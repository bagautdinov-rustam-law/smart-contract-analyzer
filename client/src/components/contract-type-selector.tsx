import { FileText, Handshake, Wrench } from "lucide-react";

export type ContractType = 'supply' | 'services' | 'works';

export interface ContractTypeConfig {
  label: string;
  description: string;
  icon: typeof FileText;
  partyA: string;      // "сильная" сторона (покупатель / заказчик)
  partyB: string;      // "слабая" сторона (поставщик / исполнитель / подрядчик)
  partyAEmoji: string;
  partyBEmoji: string;
  heroTitle: string;
}

export const CONTRACT_TYPE_CONFIG: Record<ContractType, ContractTypeConfig> = {
  supply: {
    label: 'Поставка',
    description: 'Договор поставки товаров',
    icon: FileText,
    partyA: 'Покупатель',
    partyB: 'Поставщик',
    partyAEmoji: '👤',
    partyBEmoji: '🏢',
    heroTitle: 'AI проверка договоров поставки',
  },
  services: {
    label: 'Оказание услуг',
    description: 'Договор возмездного оказания услуг',
    icon: Handshake,
    partyA: 'Заказчик',
    partyB: 'Исполнитель',
    partyAEmoji: '👤',
    partyBEmoji: '🛠️',
    heroTitle: 'AI проверка договоров оказания услуг',
  },
  works: {
    label: 'Выполнение работ',
    description: 'Договор подряда (выполнение работ)',
    icon: Wrench,
    partyA: 'Заказчик',
    partyB: 'Подрядчик',
    partyAEmoji: '👤',
    partyBEmoji: '🏗️',
    heroTitle: 'AI проверка договоров подряда',
  },
};

interface ContractTypeSelectorProps {
  contractType: ContractType;
  onContractTypeChange: (type: ContractType) => void;
}

export function ContractTypeSelector({ contractType, onContractTypeChange }: ContractTypeSelectorProps) {
  const types: ContractType[] = ['supply', 'services', 'works'];

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <FileText className="hf-orange-text mr-2" size={20} />
        Тип договора
      </h3>

      <div className="space-y-2">
        {types.map((type) => {
          const config = CONTRACT_TYPE_CONFIG[type];
          const isActive = contractType === type;
          const Icon = config.icon;

          return (
            <button
              key={type}
              onClick={() => onContractTypeChange(type)}
              className={`w-full px-4 py-3 rounded-lg border-2 transition-all duration-200 text-left flex items-center space-x-3 ${
                isActive
                  ? 'border-orange-400 bg-orange-50 shadow-sm'
                  : 'border-gray-200 bg-white hover:border-orange-200 hover:bg-orange-50/50'
              }`}
            >
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                isActive ? 'hf-orange-bg' : 'bg-gray-100'
              }`}>
                <Icon size={16} className={isActive ? 'text-white' : 'text-gray-500'} />
              </div>
              <div className="min-w-0">
                <div className={`font-medium text-sm ${isActive ? 'text-orange-900' : 'text-gray-700'}`}>
                  {config.label}
                </div>
                <div className={`text-xs truncate ${isActive ? 'text-orange-600' : 'text-gray-400'}`}>
                  {config.partyA} / {config.partyB}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
