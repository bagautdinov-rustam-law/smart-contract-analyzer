import { useState, useRef, useEffect } from "react";
import { File, ChartLine } from "lucide-react";
import { ContractInput } from "@/components/contract-input";
import { RequirementsInput } from "@/components/requirements-input";
import { RiskInput } from "@/components/risk-input";
import { AnalysisResults } from "@/components/analysis-results";
import { AnalysisProgress } from "@/components/analysis-progress";
import { StructuralAnalysisComponent } from "@/components/structural-analysis";
import { Button } from "@/components/ui/button";
import { useDeepseekAnalysis } from "@/hooks/use-deepseek-analysis";
import { exportToDocx } from "@/lib/docx-export";
import type { ContractParagraph, Contradiction, RightsImbalance, StructuralAnalysis } from "@shared/schema";
import { ContradictionsResults } from "@/components/contradictions-results";
import { RightsImbalanceResults } from "@/components/rights-imbalance-results";
import { AnalysisPerspective, PerspectiveSelector } from "@/components/perspective-selector";
import { ContractTypeSelector, type ContractType, CONTRACT_TYPE_CONFIG } from "@/components/contract-type-selector";
import { CONTRACT_PROMPTS } from "@/lib/contract-prompts";
import { TableOfContents } from "@/components/table-of-contents";
import { FloatingFilters } from "@/components/floating-filters";
import { QualityFeedback } from "@/components/quality-feedback";

// Корпоративные требования и риски загружаются из client/src/lib/contract-prompts.ts
// в зависимости от типа договора (поставка / услуги / подряд) и перспективы анализа.

const defaultContractText = `
Договор поставки

1.	Предмет Договора

1.1.	Поставщик обязуется осуществить поставку сыпучих строительных материалов (далее – Товар) на условиях, предусмотренных настоящим Договором, а Покупатель обязуется принять и оплатить Товар.
1.2.	Наименование, ассортимент, количество и цена Товара, сроки поставки, наименование грузополучателя (при наличии), адрес места назначения доставки Товара (место поставки Товара), условия оплаты и иные условия поставки определяются на основании подписанных Сторонами Спецификаций, являющихся неотъемлемой частью настоящего Договора.
1.3.	В случае расхождений между условиями Договора и условиями Спецификации применяются условия, согласованные в Спецификации.

2.	Порядок поставки

2.1.	Оформление Спецификации инициируется Поставщиком по заявке Покупателя. Заявка направляется Поставщику в произвольной письменной форме по электронной почте или телефону (в т.ч. с помощью мессенджеров), с указанием наименования Товара, необходимого количества и предполагаемых сроков поставки, иных пожеланий Покупателя. Вопросы, связанные с подготовкой Спецификации, могут согласовываться в рабочем порядке, в т.ч. по телефону. На основе заявки Покупателя Поставщик готовит Спецификацию, которую направляет Покупателю для подписания и оформления. Подписание Спецификации налагает на Стороны взаимные обязательства, подлежащие исполнению.
2.2.	Если иное не предусмотрено Спецификацией, то поставка Товара организуется Поставщиком до места назначения, указанного Покупателем.
2.3.	Товар доставляется Покупателю или грузополучателю, указанному Покупателем, автомобильным транспортом.
2.4.	Количество (масса) Товара определяется взвешиванием гружёного Товаром транспортного средства на весах, установленных на карьере. Количество Товара указывается в товарно- транспортной накладной или транспортной накладной.
2.5.	На каждое автотранспортное средство, задействованное в доставке Товара до Покупателя/грузополучателя оформляется товарно-транспортная накладная или транспортная накладная, на усмотрение Поставщика (далее по тексту «ТН», «транспортная накладная»).
2.6.	Поставка Товара осуществляется Поставщиком путем доставки Товара на склад/до места назначения Покупателя/грузополучателя. Датой поставки считается дата доставки Товара до места назначения Покупателя/грузополучателя и подписания представителем Покупателя/грузополучателя транспортной накладной. В целях реализации настоящего пункта Стороны устанавливают, что лицо, осуществляющее приёмку Товара в месте нахождения Покупателя/грузополучателя, является лицом, имеющим все необходимые полномочия для приёмки Товара и подписания транспортной накладной.
2.7.	Право собственности на Товар, поставляемый по настоящему Договору, а также риск случайной гибели или повреждения Товара переходят от Поставщика к Покупателю в момент подписания уполномоченным представителем Покупателя/грузополучателя транспортной накладной, подтверждающей приемку Товара. Подписание транспортной накладной подтверждает приемку Покупателем Товара по количеству, качеству и ассортименту.
2.8.	Покупатель обязан создать условия для правильной и своевременной приемки Товара, в том числе устройство и содержание в надлежащем состоянии подъездных дорог к месту разгрузки Товара, подготовка места разгрузки (площадки) Товара для его незамедлительной выгрузки, во избежание простоя автотранспортных средств Поставщика.
2.9.	Еженедельно, если иное не согласовано Сторонами в Спецификации, Поставщик предоставляет Покупателю универсальный передаточный документ (УПД) в двух экземплярах, в котором отражается объем и стоимость поставленного по транспортным накладным Товара в течение предыдущей недели. Покупатель обязан подписать полученный УПД и вернуть один экземпляр УПД Поставщику в течение трёх рабочих дней с даты получения от Поставщика. В случае обнаружения неточностей в оформлении УПД Покупатель уведомляет об этом Поставщика, и Стороны оформляют уточнённый УПД.
2.10.	Покупатель несёт ответственность за подготовку своего объекта к приёмке автотранспорта Поставщика с Товаром, - в частности, должны быть подготовлены подъездные пути к объекту, площадка для выгрузки Товара, площадка для разворота автотранспорта. Покупатель обязан в течение двух дней с даты получения обращения оплатить расходы Поставщика в связи неготовностью объекта Покупателя к приемке и выгрузке грузового автотранспорта (например, расходы на вытягивание увязшего в грунте или снегу автотранспорта, расходы на ремонт поломки/повреждения автотранспорта в связи с неготовностью объекта, иные расходы по устранению ситуации на объекте и ремонту автотранспорта, и т.д.).

3.	Приемка Товара по качеству и количеству

3.1.	Поставщик гарантирует, что поставляемый Товар соответствует качеству, предъявляемому законом к данному Товару. Качество Товара предполагается надлежащим, пока Покупателем не доказано иное.
3.2.	Покупатель уведомлен, что действующее законодательство Российской Федерации и технические регламенты не предусматривают норм, устанавливающих гарантийный срок на Товар. Покупатель уведомлен, что качество Товара может быть напрямую связано с условиями хранения Товара у Покупателя.
3.3.	Покупатель вправе самостоятельно и за свой счет проверить качество Товара до подписания Спецификации посредством осуществления лабораторного испытания Товара на карьере.
3.4.	Приемка Товара по количеству при поставке Товара на условиях доставки осуществляется Покупателем или грузополучателем по месту назначения доставки Товара.
Факт приемки Товара по количеству подтверждается путём подписания Покупателем или его грузополучателем транспортной накладной. Транспортная накладная подписывается непосредственно после разгрузки транспортного средства Поставщика.
3.5.	В случае, если при приемке Товара по месту назначения Покупателя/грузополучателя у Покупателя возникнут возражения по количеству поставленного Товара, Покупатель обязан сделать отметку во всех экземплярах ТН о наличии расхождений по количеству, с указанием реального объёма, и незамедлительно вызвать представителя Поставщика для дальнейшего участия в приемке Товара и оформления акта об установленных расхождениях по количеству при приемке Товара по транспортной накладной.
В случае неявки представителя Поставщика в указанный срок, Покупатель имеет право в одностороннем порядке составить акт о несоответствии количества поставленного Товара количеству, указанному в ТН. Акт о несоответствии составляется применением видеосъёмки, позволяющей оценить доводы Покупателя.
3.6.	В случае, если Покупатель/грузополучатель необоснованно отказывается от приемки Товара в момент фактической отгрузки Товара, Покупатель обязан компенсировать Поставщику расходы, связанные с доставкой Товара.
 
3.7.	Претензии по качеству Товара могут предъявляться при условии соблюдения Покупателем условий приемки и хранения щебня в соответствии с ГОСТ 8267-93. Покупатель обязан не допускать засорение, загрязнение и намокание щебня. Обязанность доказывания надлежащего хранения щебня с момента выгрузки до момента предъявления претензии по качеству лежит на Покупателе.

4.	Цена Товара. Порядок расчетов

4.1.	Цена за Товар и порядок расчетов согласовываются Сторонами в Спецификациях, являющихся неотъемлемой частью настоящего Договора.
4.2.	Оплата считается произведенной в момент поступления денежных средств на расчётный счёт Поставщика.
4.3.	Если иное не предусмотрено Спецификацией, Товар поставляется при условии внесения 100% предоплаты. В случае невнесения предоплаты Поставщик вправе приостановить отгрузку Товара, и в таком случае Поставщик не считается просрочившим поставку.
4.4.	Покупатель согласен с тем, что в случае усиления весового контроля на автодорогах по сезонным причинам, или по усмотрению органов власти, загрузка автотранспорта Товаром может быть уменьшена, при этом стоимость доставки Товара по усмотрению Поставщика может быть увеличена в одностороннем внесудебном порядке на период действия весового контроля.

5.	Ответственность сторон

5.1.	В случаях ненадлежащего исполнения обязательств по настоящему Договору стороны несут ответственность в соответствии с действующим законодательством РФ.
5.2.	За несвоевременную поставку Товара Поставщиком Покупатель вправе начислить и потребовать оплаты неустойки в размере 0,1% от стоимости не поставленного в срок Товара за каждый день просрочки, но не более 10% от стоимости не поставленного в срок Товара. В случае поставки по предоплате неустойка на неоплаченный и не поставленный в срок Товар не начисляется и не выплачивается.
5.3.	При несоблюдении согласованного в Спецификации срока платежей, Поставщик вправе начислить Покупателю пеню в размере 1% от суммы задолженности за каждый день просрочки, но не более 100% от суммы задолженности.
5.4.	За простой автотранспорта Поставщика в ожидании разгрузки в месте назначения Покупателя свыше одного часа, Поставщик вправе потребовать оплаты штрафа в размере 2 000 рублей за каждый час ожидания.
5.5.	Уплата неустоек, а также возмещение убытков, не освобождает Стороны от исполнения своих обязательств в натуре.
5.6.	В случае нарушения одной из Сторон условий Договора, Сторона, обнаружившая соответствующее нарушение, направляет претензию другой Стороне с требованием об устранении нарушений, а также об оплате неустоек, штрафов, в том числе компенсации убытков, причиненных допущенными нарушениями. Штрафные санкции подлежат уплате виновной стороной в течение пяти рабочих дней с даты получения соответствующего требования (претензии). Уплата штрафных санкций не освобождает виновную сторону от выполнения принятых на себя обязательств или устранения нарушений.
5.7.	Доставка претензии производится заказным или ценным письмом Почтой России, или курьером под расписку о получении. Стороны вправе продублировать отправку претензии по электронной почте.
5.9.	В целях урегулирования спора Стороны предусматривают обязательный претензионный порядок регулирования спора. Срок направления ответа на претензию – 10 рабочих дней с даты получения претензии.

6.	Обстоятельства непреодолимой силы
 
6.1.	Сторона, не исполнившая или ненадлежащим образом исполнившая обязательства по настоящему Договору, не несет ответственности, если докажет, что надлежащее исполнение оказалось невозможным вследствие возникновения обстоятельств непреодолимой силы (форс- мажор).
6.2.	Под обстоятельствами непреодолимой силы подразумеваются: войны, наводнения, пожары, землетрясения и прочие стихийные бедствия, забастовки, изменения действующего законодательства или любые другие обстоятельства, на которые затронутая ими Сторона не может реально воздействовать и которые она не могла разумно предвидеть, и при этом они не позволяют исполнить обязательства по настоящему договору, и возникновение которых не явилось прямым или косвенным результатом действия или бездействия одной из Сторон.
6.3.	Сторона, не исполнившая обязательства по настоящему договору в силу возникновения обстоятельств непреодолимой силы, обязана в течение 5 (пяти) рабочих дней с момента наступления обстоятельств непреодолимой силы проинформировать об этом другую сторону в письменной форме. Такая информация должна содержать данные о характере обстоятельств непреодолимой силы, а также, по возможности, оценку их влияния на исполнение и возможный срок исполнения обязательств.
6.4.	По прекращении действия указанных обстоятельств потерпевшая сторона должна незамедлительно направить письменное уведомление об этом другой стороне с указанием срока, в который предполагается исполнить обязательства по настоящему договору.
6.5.	В случае возникновения обстоятельств непреодолимой силы срок исполнения обязательств по настоящему договору продлевается на срок действия обстоятельств непреодолимой силы и их последствий.
6.6.	В том случае, если обстоятельства непреодолимой силы препятствуют одной из Сторон выполнить её обязательства в течение срока, превышающего более 3 месяцев, любая из сторон может направить другой стороне уведомление с предложением о проведении переговоров с целью определения взаимоприемлемых условий выполнения обязательств по договору или прекращения его действия.
6.7.	Стороны не освобождаются от выполнения обязательств, срок выполнения которых наступил до возникновения обстоятельств непреодолимой силы.
6.8.	Надлежащим доказательством наличия обстоятельств непреодолимой силы и их продолжительности будут служить документы, выданные соответствующими государственными органами.

7.	Порядок изменения и расторжения настоящего Договора

7.1.	Условия настоящего Договора имеют одинаковую юридическую силу для Сторон Договора.
7.2.	Любые изменения и дополнения к настоящему Договору действительны при условии, если они совершены в письменной форме и подписаны надлежаще уполномоченными на то представителями Сторон.
7.3.	При изменении реквизитов, а также в случаях реорганизации и ликвидации Стороны обязаны в течение 5 (пяти) дней уведомить друг друга о произошедших изменениях.
В случае не извещения (несвоевременного извещения) об изменении адресов все уведомления, направленные по адресам, указанные в настоящем Договоре, считаются надлежащим уведомлением Сторон.
7.4.	Настоящий Договор может быть изменен или расторгнут по соглашению Сторон. Стороны должны выполнить все свои обязательства по Договору, возникшие до момента прекращения его действия.
Сторона, которой направлено предложение о расторжении настоящего Договора по соглашению Сторон, должна дать письменный ответ по существу в срок не позднее 5 (пяти) рабочих дней со дня его получения. Расторжение настоящего Договора производится путем подписания Сторонами соответствующего соглашения о расторжении.
7.5.	Покупатель вправе в одностороннем внесудебном порядке расторгнуть настоящий Договор в случае неоднократного нарушения Поставщиком сроков поставки.
 
7.6.	Поставщик вправе в одностороннем внесудебном порядке расторгнуть настоящий Договор в случаях:
-	неоднократного нарушения сроков оплаты Товара Покупателем,
-	неоднократной невыборки объёмов Товара Покупателем.
7.7.	Уведомление о расторжении Договора (отказе от Договора) направляется заинтересованной Стороной в адрес другой Стороны заказным или ценным письмом Почтой России, при этом Стороны вправе продублировать отправку по электронной почте.
7.8.	Любая из Сторон по собственной инициативе вправе расторгнуть в одностороннем внесудебном порядке Договор, при условии уведомления другой Стороны не менее, чем за 15 дней до даты расторжения. При этом все обязательства, возникшие до даты расторжения Договора, подлежат исполнению Сторонами.
7.9.	Стороны исходят из того, что прекращение действия настоящего Договора по любой причине должно сводить к минимальному ущербу для Сторон.

8.	Срок действия Договора

8.1.	Настоящий Договор вступает в силу с момента подписания его Сторонами и действует до
«31» декабря 2024 года, а в части обязательств, принятых сторонами до окончания срока действия Договора, - до их полного надлежащего исполнения Сторонами.
8.2.	В случае, если ни одна из Сторон до окончания срока действия Договора не заявила о его расторжении, то действие Договора возобновляется на аналогичный период на тех же условиях.

`;

export default function ContractAnalyzer() {
    const [contractText, setContractText] = useState("");
    const [contractType, setContractType] = useState<ContractType>('supply');
    const [perspective, setPerspective] = useState<AnalysisPerspective>('buyer');
    const [checklistText, setChecklistText] = useState(CONTRACT_PROMPTS.supply.buyer.checklist);
    const [riskText, setRiskText] = useState(CONTRACT_PROMPTS.supply.buyer.risks);
    const [contractParagraphs, setContractParagraphs] = useState<ContractParagraph[]>([]);
    const [missingRequirements, setMissingRequirements] = useState<ContractParagraph[]>([]);
    const [ambiguousConditions, setAmbiguousConditions] = useState<ContractParagraph[]>([]);
    const [structuralAnalysis, setStructuralAnalysis] = useState<StructuralAnalysis | undefined>(undefined);
    const [contradictions, setContradictions] = useState<Contradiction[]>([]);
    const [rightsImbalance, setRightsImbalance] = useState<RightsImbalance[]>([]);
    const [showCompliance, setShowCompliance] = useState(true);
    const [showPartial, setShowPartial] = useState(true);
    const [showRisks, setShowRisks] = useState(true);
    const [showMissing, setShowMissing] = useState(true);
    const [showOther, setShowOther] = useState(true);
    const [showContradictions, setShowContradictions] = useState(true);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    const {
        analyzeContract,
        isLoading,
        error,
        progress,
    } = useDeepseekAnalysis();

    // Ссылка на раздел структурного анализа для автоматической прокрутки
    const structuralAnalysisRef = useRef<HTMLDivElement>(null);

    // Автоматическая прокрутка к структурному анализу после завершения анализа
    useEffect(() => {
        if (structuralAnalysis && !isAnalyzing && structuralAnalysisRef.current) {
            // Небольшая задержка для завершения рендеринга
            setTimeout(() => {
                if (structuralAnalysisRef.current) {
                    // Получаем позицию элемента
                    const elementTop = structuralAnalysisRef.current.offsetTop;
                    // Вычитаем высоту заголовка (64px) + дополнительный отступ (24px)
                    const offsetTop = elementTop - 88;
                    
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });
                }
            }, 300);
        }
    }, [structuralAnalysis, isAnalyzing]);


    const handleContractTypeChange = (newType: ContractType) => {
        setContractType(newType);
        setPerspective('buyer');
        setContractParagraphs([]);
        setMissingRequirements([]);
        setAmbiguousConditions([]);
        setStructuralAnalysis(undefined);
        setContradictions([]);
        setRightsImbalance([]);
        // Загружаем чек-листы и риски для нового типа договора
        setChecklistText(CONTRACT_PROMPTS[newType].buyer.checklist);
        setRiskText(CONTRACT_PROMPTS[newType].buyer.risks);
    };

    const handlePerspectiveChange = (newPerspective: 'buyer' | 'supplier') => {
        setPerspective(newPerspective);
        // Сбрасываем результаты анализа при смене перспективы
        setContractParagraphs([]);
        setMissingRequirements([]);
        setAmbiguousConditions([]);
        setStructuralAnalysis(undefined);
        setContradictions([]);
        setRightsImbalance([]);
        
        // Обновляем чек-лист и риски в зависимости от перспективы и типа договора
        const prompts = CONTRACT_PROMPTS[contractType][newPerspective];
        setChecklistText(prompts.checklist);
        setRiskText(prompts.risks);
    };

    const handleAnalyze = async () => {
        if (!contractText.trim() || !checklistText.trim()) {
            alert("Пожалуйста, заполните текст договора и чек-лист");
            return;
        }

        setIsAnalyzing(true);
        
        try {
            const { contractParagraphs, missingRequirements, ambiguousConditions, structuralAnalysis, contradictions, rightsImbalance } = await analyzeContract(
                contractText,
                checklistText,
                riskText,
                perspective,
                (progressMessage: string) => {
                    console.log("Progress:", progressMessage);
                }
            );
            
            setContractParagraphs(contractParagraphs || []);
            setMissingRequirements(missingRequirements || []);
            setAmbiguousConditions(ambiguousConditions || []);
            setStructuralAnalysis(structuralAnalysis || undefined);
            setContradictions(contradictions || []);
            setRightsImbalance(rightsImbalance || []);
            
            // Успешное завершение - останавливаем анализ
            setIsAnalyzing(false);
        } catch (error) {
            console.error("Analysis failed:", error);
            setIsAnalyzing(false);
        }
    };

    const handleAnalysisComplete = () => {
        // Прогресс-бар сам управляет своим состоянием
        // Ничего не делаем здесь
    };

    const handleUpdateComment = (id: string, newComment: string) => {
        setContractParagraphs(prev => 
            prev.map(paragraph => 
                paragraph.id === id 
                ? { ...paragraph, comment: newComment }
                : paragraph
            )
        );
        
        setMissingRequirements(prev => 
            prev.map(requirement => 
                requirement.id === id 
                ? { ...requirement, comment: newComment }
                : requirement
            )
        );
        
        setAmbiguousConditions(prev => 
            prev.map(condition => 
                condition.id === id 
                ? { ...condition, comment: newComment }
                : condition
            )
        );
    };

    const handleSubmitFeedback = (feedback: any) => {
        console.log('Feedback received:', feedback);
        // Здесь можно отправить отзыв на сервер для дальнейшего анализа
    };

    const loadExample = () => {
        setContractText(defaultContractText);
    };

    // Добавляем проверку на undefined
    const filteredResults = (contractParagraphs || []).filter((paragraph) => {
        if (!paragraph.category) return true;
        if (paragraph.category === 'checklist' && !showCompliance) return false;
        if (paragraph.category === 'partial' && !showPartial) return false;
        if (paragraph.category === 'risk' && !showRisks) return false;
        if (paragraph.category === 'missing' && !showMissing) return false;
        if ((paragraph.category === 'other' || paragraph.category === 'ambiguous') && !showOther) return false;
        return true;
    });

    // Добавляем проверку на undefined
    const complianceCount = (contractParagraphs || []).filter(p => p.category === 'checklist').length;
    const partialCount = (contractParagraphs || []).filter(p => p.category === 'partial').length;
    const riskCount = (contractParagraphs || []).filter(p => p.category === 'risk').length;
    const missingCountFromParagraphs = (contractParagraphs || []).filter(p => p.category === 'missing').length;
    const missingCountTotal = missingCountFromParagraphs + (missingRequirements || []).length;
    const otherCount = (contractParagraphs || []).filter(p => p.category === 'other' || p.category === 'ambiguous').length;

    return (
        <div className="min-h-screen bg-white">
            {/* Header */}
            <header className="sticky top-0 z-50 bg-white border-b border-gray-200 shadow-sm">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-16">
                        <div className="flex items-center space-x-3">
                            <div className="w-8 h-8 hf-orange-bg rounded-lg flex items-center justify-center">
                                <File className="text-white text-sm" size={16} />
                            </div>
                            <div>
                                <h1 className="text-xl font-semibold text-gray-900">AI Скрепка</h1>
                            </div>
                        </div>
                        <div className="flex items-center space-x-4">
                            <Button
                                variant="outline"
                                onClick={() => window.open('/analytics', '_blank')}
                                className="text-sm"
                            >
                                Аналитика качества
                            </Button>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                                <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                                <span>AI API подключен</span>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Hero Section */}
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-gray-900 mb-4">{CONTRACT_TYPE_CONFIG[contractType].heroTitle}</h2>
                    <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                        Установите корпоративные критерии, риски и обязательные условия.
                    </p>
                    <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                        Сервис проконтролирует соответствие каждого договора вашим стандартам.
                    </p>
                </div>

                <div className="space-y-8">
                    {/* Input Panel - Full Width */}
                    <div className="space-y-6">
                        {/* Contract Input and Perspective Selection in parallel */}
                        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                            <div className="lg:col-span-7">
                                <ContractInput
                                    value={contractText}
                                    onChange={setContractText}
                                />
                            </div>
                            <div className="lg:col-span-2">
                                <ContractTypeSelector
                                    contractType={contractType}
                                    onContractTypeChange={handleContractTypeChange}
                                />
                            </div>
                            <div className="lg:col-span-3">
                                <PerspectiveSelector
                                    perspective={perspective}
                                    onPerspectiveChange={handlePerspectiveChange}
                                    contractType={contractType}
                                />
                            </div>
                        </div>

                        {/* Requirements and Risks in parallel */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <RequirementsInput
                                value={checklistText}
                                onChange={setChecklistText}
                                perspective={perspective}
                                perspectiveLabel={perspective === 'buyer' ? CONTRACT_TYPE_CONFIG[contractType].partyA : CONTRACT_TYPE_CONFIG[contractType].partyB}
                            />
                            <RiskInput
                                value={riskText}
                                onChange={setRiskText}
                                perspective={perspective}
                                perspectiveLabel={perspective === 'buyer' ? CONTRACT_TYPE_CONFIG[contractType].partyA : CONTRACT_TYPE_CONFIG[contractType].partyB}
                            />
                        </div>

                        {/* Analysis Button */}
                        <div className="flex flex-col items-center mt-6">
                            <Button
                                onClick={handleAnalyze}
                                disabled={!contractText.trim() || isLoading || isAnalyzing}
                                className="hf-orange-bg hover:hf-orange-bg text-white px-8 py-4 h-auto text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-200"
                            >
                                <ChartLine className="mr-2" size={20} />
                                {isAnalyzing ? 'Анализируем...' : 'Анализировать договор'}
                                <File className="ml-2" size={20} />
                            </Button>
                        </div>

                        {/* Examples Section */}
                        <div className="mt-6 bg-white rounded-lg shadow-sm border border-gray-200 p-3">
                            <Button
                                onClick={loadExample}
                                variant="ghost"
                                className="w-full p-3 h-auto text-left hover:bg-gray-50 transition-all rounded-lg"
                            >
                                <div className="flex items-center">
                                    <div className="w-8 h-8 bg-blue-50 rounded-lg flex items-center justify-center mr-3">
                                        <File className="text-blue-600" size={16} />
                                    </div>
                                    <div>
                                        <div className="font-medium text-gray-900">Попробуйте пример договора</div>
                                        <div className="text-sm text-gray-500">Стандартный договор поставки товаров (пример)</div>
                                    </div>
                                </div>
                            </Button>
                        </div>
                    </div>


                    {/* Error Display */}
                    {error && (
                        <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-6">
                            <div className="flex items-center">
                                <div className="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center mr-3">
                                    <span className="text-red-600 text-sm">⚠</span>
                                </div>
                                <div>
                                    <h3 className="text-lg font-semibold text-red-800">Ошибка анализа</h3>
                                    <p className="text-red-600 mt-1">{error}</p>
                                </div>
                            </div>
                        </div>
                    )}
                    
                    {/* Results Layout with Table of Contents */}
                    {(structuralAnalysis || contractParagraphs.length > 0) && (
                        <div className="mt-6 grid grid-cols-1 lg:grid-cols-4 gap-8">
                            {/* Main Content - narrower */}
                            <div className="lg:col-span-3 space-y-6">
                                {/* Structural Analysis */}
                                {structuralAnalysis && (
                                    <div id="structural-analysis" ref={structuralAnalysisRef}>
                                        <StructuralAnalysisComponent analysis={structuralAnalysis} />
                                    </div>
                                )}
                                
                                {/* Results */}
                                {contractParagraphs.length > 0 && (
                                    <div id="analysis-results">
                                        <AnalysisResults
                                            contractParagraphs={contractParagraphs}
                                            missingRequirements={missingRequirements}
                                            ambiguousConditions={ambiguousConditions}
                                            showCompliance={showCompliance}
                                            showPartial={showPartial}
                                            showRisks={showRisks}
                                            showOther={showOther}
                                            showMissing={showMissing}
                                            exportToDocx={() => exportToDocx({
                                                contractParagraphs,
                                                missingRequirements,
                                                contradictions,
                                                rightsImbalance,
                                                structuralAnalysis
                                            })}
                                            onUpdateComment={handleUpdateComment}
                                            onSubmitFeedback={handleSubmitFeedback}
                                            showContradictions={showContradictions}
                                            onToggleCompliance={setShowCompliance}
                                            onTogglePartial={setShowPartial}
                                            onToggleRisks={setShowRisks}
                                            onToggleMissing={setShowMissing}
                                            onToggleOther={setShowOther}
                                            onToggleContradictions={setShowContradictions}
                                            complianceCount={complianceCount}
                                            partialCount={partialCount}
                                            riskCount={riskCount}
                                            missingCount={missingCountTotal}
                                            otherCount={otherCount}
                                            contradictionsCount={contradictions.length}
                                        />
                                    </div>
                                )}

                                {/* Отсутствующие требования — якорь в компоненте AnalysisResults */}
                                
                                {/* Contradictions Results */}
                                <div id="contradictions-results">
                                    <ContradictionsResults
                                        contradictions={contradictions}
                                        showContradictions={showContradictions}
                                    />
                                </div>

                                {/* Rights Imbalance Results */}
                                <div id="rights-imbalance-results">
                                    <RightsImbalanceResults
                                        rightsImbalance={rightsImbalance}
                                    />
                                </div>
                            </div>

                            {/* Table of Contents and Filters - Right Sidebar */}
                            <div className="lg:col-span-1">
                                <div className="sticky top-24 space-y-6">
                                    <TableOfContents
                                        hasStructuralAnalysis={!!structuralAnalysis}
                                        hasResults={contractParagraphs.length > 0}
                                        hasContradictions={contradictions.length > 0}
                                        hasRightsImbalance={rightsImbalance.length > 0}
                                        hasMissingRequirements={missingRequirements.length > 0}
                                    />
                                    {/* Фильтр — отдельный блок под оглавлением */}
                                    {contractParagraphs.length > 0 && (
                                        <FloatingFilters
                                            showCompliance={showCompliance}
                                            showPartial={showPartial}
                                            showRisks={showRisks}
                                            showMissing={showMissing}
                                            showOther={showOther}
                                            showContradictions={showContradictions}
                                            onToggleCompliance={setShowCompliance}
                                            onTogglePartial={setShowPartial}
                                            onToggleRisks={setShowRisks}
                                            onToggleMissing={setShowMissing}
                                            onToggleOther={setShowOther}
                                            onToggleContradictions={setShowContradictions}
                                            complianceCount={complianceCount}
                                            partialCount={partialCount}
                                            riskCount={riskCount}
                                            missingCount={missingCountTotal}
                                            otherCount={otherCount}
                                            contradictionsCount={contradictions.length}
                                        />
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    <AnalysisProgress
                        isAnalyzing={isAnalyzing}
                        onComplete={handleAnalysisComplete}
                        progress={progress}
                    />
                    {/* Оцените качество анализа — всегда в самом конце */}
                    {(structuralAnalysis || contractParagraphs.length > 0) && (
                        <div className="mt-12">
                            <QualityFeedback 
                                analysisId={`analysis_${Date.now()}`}
                                onSubmitFeedback={handleSubmitFeedback}
                            />
                        </div>
                    )}
                </div>
            </div>
            {/* Footer */}
            <footer className="bg-white border-t border-gray-200 mt-16">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    <div className="text-center">
                        <p className="text-sm text-gray-600">
                            Проект телеграм-канала{' '}
                            <a
                                href="https://t.me/+plgepYs_y0M3ODJi"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="hf-orange-text hover:text-orange-700 font-medium transition-colors"
                            >
                                "AI Скрепка: Связь Права и Технологий"
                            </a>
                        </p>
                        <p className="text-xs text-gray-500 mt-2">
                            Разработчик: Мирошниченко Евгений — старший юрист в консалтинге и энтузиаст новых технологий
                        </p>
                    </div>
                </div>
            </footer>
        </div> // ИСПРАВЛЕНИЕ: Закрывающий тег для корневого <div className="min-h-screen bg-gray-50">
    );
}
