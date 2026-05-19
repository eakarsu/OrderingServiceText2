import OrderKanban from '../components/OrderKanban.js';
import DailyVolumeChart from '../components/DailyVolumeChart.js';
import SMSBroadcast from '../components/SMSBroadcast.js';
import ReceiptPDF from '../components/ReceiptPDF.js';

export default function CustomViewsPage() {
  return (
    <div data-testid="custom-views-page" className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Order Views</h1>
        <p className="text-sm text-gray-600">
          Operational tools: kanban board, daily volume, SMS broadcasts, and receipt PDFs.
        </p>
      </div>

      <OrderKanban />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DailyVolumeChart />
        <ReceiptPDF />
      </div>

      <SMSBroadcast />
    </div>
  );
}
