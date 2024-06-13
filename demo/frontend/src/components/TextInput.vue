<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useUserStore } from '@/stores/user'
import { storeToRefs } from 'pinia'

const userStore = useUserStore()
const { tasks, inputText } = storeToRefs(userStore)

const noTask = computed(() => tasks.value.length === 0)

const selectedExample = ref()
const examples = ref([
  {
    name: 'Patientin mit Brustschmerz',
    text: 'Die Patientin Elizabeth Brönnimann hatte sich heute in der Notfallaufnahme gemeldet weil sie Brustschmerzen hatte. Es wurde ebenfalls ein Röntgenbild erstellt, dabei zeigte sich aber keine Lungenentzündung.'
  },
  {
    name: 'Grössenprogrediente Läsion',
    text: 'CT Thorax-Abdomen: Grössenprogrediente Läsion im Segment III festgestellt. Zeigt CT-morphologisch keine klassische KM-Dynamik, vorerst als HCC zu werten und weitere Abklärung durch MRT benötigt.'
  },
  {
    name: 'Medikation',
    text: 'Der Patientin wurde 2x täglich Xarelto 20mg und Pantozol 40mg verschrieben.'
  },
  {
    name: 'Notfallgespräch',
    text:
      'Sprecher D: Alle parat für die Übergabe?\n' +
      'Sprecher D: [PAUSE] Das ist Franz Feldmann, 66 Jahre. Hat vernichtende Brustschmerzen seit etwa 30 Minuten. Strahlen aus in den linken Arm. Hat etwas Dyspnö, ist nicht synkopiert. Keine KHK bekannt. Hat noch Hypertonus, kennt aber seine Medis nicht. Vitalparameter waren Blutdruck 145 zu 90, Puls 95, Sauerstoffsättigung 93% an Raumluft. Haben mal 4 Liter Sauerstoff und 2mg Morphin gegeben. Das EKG hatten wir Euch schon per Mail geschickt. Da hat es Senkungen in V2 bis V4. Wir haben mit Heparin noch gewartet, weil er nicht sicher war, ob er auch was zur Blutverdünnung nimmt.\n' +
      'Sprecher B: Vielen Dank. Peter, kannst Du bitte das EKG organisieren?\n' +
      'Sprecher C: Klar, mache ich.\n' +
      'Sprecher B: [PAUSE] Guten Tag Herr Feldmann, ich bin Dr. Fischer. Sie haben Schmerzen in der Brust?\n' +
      'Sprecher A: Ja, und sie sind wirklich stark. Ich hatte in den letzten Monaten schon öfter so ein leichtes Ziehen, aber jetzt ist es viel intensiver.\n' +
      'Sprecher B: Strahlt der Schmerz in den Arm, den Hals oder den Rücken aus?\n' +
      'Sprecher A: Ja, der linke Arm tut weh und der Kiefer schmerzt.\n' +
      'Sprecher B: Haben Sie Schwierigkeiten beim Atmen oder ein Gefühl von Benommenheit?\n' +
      'Sprecher A: Vielleicht ist es auch nur die Aufregung aber ich bin etwas knapp mit der Luft..\n' +
      'Sprecher B: Nehmen Sie Medikamente?\n' +
      'Sprecher A: Ich nehme Tabletten für meinen Bluthochdruck, aber den Namen weiß ich nicht.\n' +
      'Sprecher B: Rauchen oder Alkohol?\n' +
      'Sprecher A: Nein, beides nicht.\n' +
      'Sprecher B: Gut. \n' +
      'Sprecher B: [PAUSE] Ich werde Sie jetzt untersuchen. Zunächst möchte ich Ihren Brustkorb abhören. Bitte atmen Sie tief durch.\n' +
      'Sprecher B: [PAUSE] Herz- und Lungengeräusche sind normal. Jetzt werde ich Ihren Bauch abtasten.\n' +
      'Sprecher A: Okay.\n' +
      'Sprecher B: [PAUSE] Der Bauch ist weich, nicht druckschmerzhaft. Leber und Milz nicht vergrößert. \n' +
      'Sprecher B: Peter, hast Du das EKG?\n' +
      'Sprecher C: Ja, hier.\n' +
      'Sprecher B: [PAUSE] Ja, es hat ST-Senkungen – ist aber verwackelt. Machst Du bitte ein neues?\n' +
      'Sprecher C: Mache ich!\n' +
      'Sprecher C: [PAUSE] Herr Feldmann, ich lege Ihnen jetzt die Elektroden an, dafür muss ich etwas rasieren.\n' +
      'Sprecher B: Herr Feldmann, hat jemand in Ihrer Familie Herzerkrankungen?\n' +
      'Sprecher A: Ja, mein Vater hatte einen Herzinfarkt in meinem Alter.\n' +
      'Sprecher B: Waren Sie in letzter Zeit beim Hausarzt oder Kardiologen?\n' +
      'Sprecher A: Es ist schon eine Weile her. Mein Hausarzt hat mir die Blutdrucktabletten verschrieben, aber ich war lange nicht zur Kontrolle. Ich hole die einfach von der Apotheke.\n' +
      'Sprecher B: Wie ist das mit der Blutverdünnung?\n' +
      'Sprecher A: Ich habe so etwas, was ich mir manchmal auf die Beine reibe, Heparincreme oder so wegen den Krampfadern.\n' +
      'Sprecher B: OK, dann müssen wir da nichts beachten.\n' +
      'Sprecher B: [PAUSE] Peter, machst Du bitte Herz-Schema mit Troponin, CK und pro-BNP?\n' +
      'Sprecher C: [PAUSE] In Ordnung, ich nehme jetzt Blut ab, Herr Feldmann.\n' +
      'Sprecher A: Okay.\n' +
      'Sprecher B: [PAUSE] Also auch im neuen EKG zeigen sich diese Veränderungen, die für einen Herzinfarkt sprechen. Ich melde Sie dem Kardiologen und voraussichtlich wird es eine Koronarangiografie geben.\n' +
      'Sprecher A: Was bedeutet das?\n' +
      'Sprecher B: Entschuldigung. Das ist eine Herzkatheteruntersuchung, da wird ein Schlauch in die Arterie am Handgelenk eingeführt und die Kardiologen können sich dann die Herzkranzgefässe anschauen. Wenn es dort Engstellen hat, können sie die aufweiten und mit einem Körbchen, einem Stent sichern.\n' +
      'Sprecher A: [PAUSE] Ist das gefährlich?\n' +
      'Sprecher B: Das gefährliche ist der Herzinfarkt selbst. Wenn wir nichts machen, wird eine grosse Narbe im Herzmuskel zurückbleiben. Moment bitte, der Kardiologe ruft zurück.\n' +
      'Sprecher B: [PAUSE] Ja, ja. \n' +
      'Sprecher B: [PAUSE] Das ist richtig. 66 Jahre alt. Nein, Keine Antikoagulation. Hat bisher nur Morphin. OK, dann geben wir noch Heparin. \n' +
      'Sprecher B: [PAUSE] KORO eins? Wann? Willst Du das Troponin noch abwarten\n' +
      'Sprecher B: [PAUSE] OK, dann bringen wir ihn gleich rüber!\n' +
      'Sprecher B: [PAUSE] \n' +
      'Sprecher B: [PAUSE] Peter? Wir können los. Nimmst Du bitte Sauerstoff und Notfallkoffer mit? Wir gehen ins KORO eins.\n' +
      'Sprecher C: Mache ich. Anita? Wir gehen ins KORO. Übernimmst Du bitte Koje 2 für mich?\n' +
      'Sprecher B: [PAUSE]\n' +
      'Sprecher B: [PAUSE] OK, Herr Feldmann. Ich erkläre Ihnen auf dem Weg, wie es weiter geht.\n' +
      'Sprecher A: Danke. Können Sie bitte noch meine Frau anrufen?\n' +
      'Sprecher B: Mache ich, nachdem Sie in der Koronarangiografie sind.'
  }
])

watch(
  () => selectedExample.value,
  (newVal) => {
    inputText.value = newVal.text
  }
)
</script>

<template>
  <section class="input-layout">
    <Textarea v-model="inputText" rows="5" class="text-area" :disabled="userStore.ongoingRequest" />

    <section class="text-actions">
      <div>
        <Button
          v-if="noTask"
          label="Wählen sie mindestens eine Aufhabe aus"
          icon="pi pi-ban"
          size="large"
          class="analyze-button"
          :disabled="noTask || userStore.ongoingRequest"
        />
        <Button
          v-else-if="inputText.length === 0"
          label="Für die Analyse ist Text benötigt"
          icon="pi pi-ban"
          size="large"
          class="analyze-button"
          :disabled="true"
        />
        <Button
          v-else
          @click="userStore.getAnalysis"
          label="Text Analysieren"
          icon="pi pi-microchip-ai"
          size="large"
          class="analyze-button"
          :loading="userStore.ongoingRequest"
        />
      </div>

      <Select
        v-model="selectedExample"
        :options="examples"
        optionLabel="name"
        placeholder="Beispiel auswählen"
        :loading="userStore.ongoingRequest"
        :disabled="userStore.ongoingRequest"
      />

      <p v-if="userStore.pipelineData && userStore.pipelineData?.execution_time">
        Die Analyse hat {{ userStore.pipelineData.execution_time }} Millisekunden gedauert
      </p>
    </section>
  </section>
</template>

<style scoped>
.text-area {
  width: 100%;
}

.input-layout {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.text-actions {
  display: flex;
  flex-direction: row;
  gap: 1rem;
  align-items: center;
}

.text-actions p {
  margin: 0;
}
</style>
